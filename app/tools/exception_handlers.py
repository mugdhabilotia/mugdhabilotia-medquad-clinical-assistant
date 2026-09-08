# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import random
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from app.tools.schemas import ToolExecutionResult

logger = logging.getLogger("medquad.tools.exception_handler")


class UpstreamServiceError(Exception):
    """Exception representing an upstream HTTP 500-range service error."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        response_body: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class UpstreamRateLimitError(Exception):
    """Exception representing an upstream HTTP 429 rate limit / quota error."""

    def __init__(
        self,
        message: str,
        retry_after: float = 2.0,
        response_body: str | None = None,
    ):
        super().__init__(message)
        self.status_code = 429
        self.retry_after = retry_after
        self.response_body = response_body


def resilient_function_call(
    max_retries: int = 3,
    initial_backoff_sec: float = 1.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
) -> Callable:
    """Decorator for tool functions that standardizes function calling logic,

    enforces Pydantic schema validation, and handles HTTP 429 (Rate Limit) and
    HTTP 500 (Upstream Error) exceptions natively with exponential backoff and jitter.
    """

    def decorator(func: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            start_time = time.time()
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    res = await func(*args, **kwargs)
                    if isinstance(res, ToolExecutionResult):
                        return res.model_dump()
                    if isinstance(res, dict) and "status" in res:
                        return res
                    return ToolExecutionResult(
                        status="SUCCESS",
                        data=res,
                        metadata={
                            "duration_ms": round((time.time() - start_time) * 1000, 2)
                        },
                    ).model_dump()

                except ValidationError as val_err:
                    logger.error(
                        f"Parameter validation failed in {func.__name__}: {val_err}"
                    )
                    return ToolExecutionResult(
                        status="ERROR_VALIDATION_FAILED",
                        http_code=400,
                        error=f"Strict parameter bounds violated: {val_err.errors()}",
                        retryable=False,
                        metadata={
                            "duration_ms": round((time.time() - start_time) * 1000, 2)
                        },
                    ).model_dump()

                except UpstreamRateLimitError as rate_err:
                    last_exception = rate_err
                    if attempt < max_retries:
                        delay = initial_backoff_sec * (backoff_multiplier**attempt)
                        if jitter:
                            delay += random.uniform(0.1, 0.5)
                        logger.warning(
                            f"HTTP 429 in {func.__name__}: Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue

                    return ToolExecutionResult(
                        status="ERROR_RATE_LIMITED",
                        http_code=429,
                        error=f"Upstream rate limit exceeded (HTTP 429): {rate_err}",
                        retryable=True,
                        metadata={
                            "attempts": attempt + 1,
                            "duration_ms": round((time.time() - start_time) * 1000, 2),
                        },
                    ).model_dump()

                except (UpstreamServiceError, ConnectionError) as srv_err:
                    last_exception = srv_err
                    status_code = getattr(srv_err, "status_code", 500)
                    if attempt < max_retries:
                        delay = initial_backoff_sec * (backoff_multiplier**attempt)
                        if jitter:
                            delay += random.uniform(0.1, 0.5)
                        logger.warning(
                            f"HTTP {status_code} in {func.__name__}: Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue

                    return ToolExecutionResult(
                        status="ERROR_UPSTREAM_SERVICE",
                        http_code=status_code,
                        error=f"Upstream service failure (HTTP {status_code}): {srv_err}",
                        retryable=True,
                        metadata={
                            "attempts": attempt + 1,
                            "duration_ms": round((time.time() - start_time) * 1000, 2),
                        },
                    ).model_dump()

                except Exception as unhandled_err:
                    # Check for HTTP status embedded in standard exceptions or libraries
                    err_msg = str(unhandled_err)
                    status_code = getattr(
                        unhandled_err,
                        "status_code",
                        getattr(unhandled_err, "code", None),
                    )
                    if (
                        status_code == 429
                        or "429" in err_msg
                        or "RESOURCE_EXHAUSTED" in err_msg
                    ):
                        if attempt < max_retries:
                            delay = (
                                initial_backoff_sec * (backoff_multiplier**attempt)
                            ) + random.uniform(0.1, 0.4)
                            logger.warning(
                                f"Detected 429 error, retrying attempt {attempt + 1}: {unhandled_err}"
                            )
                            await asyncio.sleep(delay)
                            continue
                        return ToolExecutionResult(
                            status="ERROR_RATE_LIMITED",
                            http_code=429,
                            error=f"HTTP 429 Rate Limit: {unhandled_err}",
                            retryable=True,
                        ).model_dump()

                    if (
                        status_code in (500, 502, 503, 504)
                        or "500" in err_msg
                        or "INTERNAL" in err_msg
                    ):
                        if attempt < max_retries:
                            delay = (
                                initial_backoff_sec * (backoff_multiplier**attempt)
                            ) + random.uniform(0.1, 0.4)
                            logger.warning(
                                f"Detected 500 error, retrying attempt {attempt + 1}: {unhandled_err}"
                            )
                            await asyncio.sleep(delay)
                            continue
                        return ToolExecutionResult(
                            status="ERROR_UPSTREAM_SERVICE",
                            http_code=500,
                            error=f"HTTP 500 Upstream Service Error: {unhandled_err}",
                            retryable=True,
                        ).model_dump()

                    logger.error(
                        f"Unhandled tool error in {func.__name__}: {unhandled_err}"
                    )
                    return ToolExecutionResult(
                        status="ERROR_INTERNAL",
                        error=f"Tool execution exception: {unhandled_err}",
                        retryable=False,
                    ).model_dump()

            return ToolExecutionResult(
                status="ERROR_RETRIES_EXHAUSTED",
                error=f"Exhausted {max_retries} attempts: {last_exception}",
                retryable=True,
            ).model_dump()

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            start_time = time.time()
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    res = func(*args, **kwargs)
                    if isinstance(res, ToolExecutionResult):
                        return res.model_dump()
                    if isinstance(res, dict) and "status" in res:
                        return res
                    return ToolExecutionResult(
                        status="SUCCESS",
                        data=res,
                        metadata={
                            "duration_ms": round((time.time() - start_time) * 1000, 2)
                        },
                    ).model_dump()

                except ValidationError as val_err:
                    logger.error(
                        f"Parameter validation failed in {func.__name__}: {val_err}"
                    )
                    return ToolExecutionResult(
                        status="ERROR_VALIDATION_FAILED",
                        http_code=400,
                        error=f"Strict parameter bounds violated: {val_err.errors()}",
                        retryable=False,
                        metadata={
                            "duration_ms": round((time.time() - start_time) * 1000, 2)
                        },
                    ).model_dump()

                except UpstreamRateLimitError as rate_err:
                    last_exception = rate_err
                    if attempt < max_retries:
                        delay = initial_backoff_sec * (backoff_multiplier**attempt)
                        if jitter:
                            delay += random.uniform(0.1, 0.5)
                        logger.warning(
                            f"HTTP 429 in {func.__name__}: Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue

                    return ToolExecutionResult(
                        status="ERROR_RATE_LIMITED",
                        http_code=429,
                        error=f"Upstream rate limit exceeded (HTTP 429): {rate_err}",
                        retryable=True,
                        metadata={
                            "attempts": attempt + 1,
                            "duration_ms": round((time.time() - start_time) * 1000, 2),
                        },
                    ).model_dump()

                except (UpstreamServiceError, ConnectionError) as srv_err:
                    last_exception = srv_err
                    status_code = getattr(srv_err, "status_code", 500)
                    if attempt < max_retries:
                        delay = initial_backoff_sec * (backoff_multiplier**attempt)
                        if jitter:
                            delay += random.uniform(0.1, 0.5)
                        logger.warning(
                            f"HTTP {status_code} in {func.__name__}: Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue

                    return ToolExecutionResult(
                        status="ERROR_UPSTREAM_SERVICE",
                        http_code=status_code,
                        error=f"Upstream service failure (HTTP {status_code}): {srv_err}",
                        retryable=True,
                        metadata={
                            "attempts": attempt + 1,
                            "duration_ms": round((time.time() - start_time) * 1000, 2),
                        },
                    ).model_dump()

                except Exception as unhandled_err:
                    err_msg = str(unhandled_err)
                    status_code = getattr(
                        unhandled_err,
                        "status_code",
                        getattr(unhandled_err, "code", None),
                    )
                    if (
                        status_code == 429
                        or "429" in err_msg
                        or "RESOURCE_EXHAUSTED" in err_msg
                    ):
                        if attempt < max_retries:
                            delay = (
                                initial_backoff_sec * (backoff_multiplier**attempt)
                            ) + random.uniform(0.1, 0.4)
                            logger.warning(
                                f"Detected 429 error, retrying attempt {attempt + 1}: {unhandled_err}"
                            )
                            time.sleep(delay)
                            continue
                        return ToolExecutionResult(
                            status="ERROR_RATE_LIMITED",
                            http_code=429,
                            error=f"HTTP 429 Rate Limit: {unhandled_err}",
                            retryable=True,
                        ).model_dump()

                    if (
                        status_code in (500, 502, 503, 504)
                        or "500" in err_msg
                        or "INTERNAL" in err_msg
                    ):
                        if attempt < max_retries:
                            delay = (
                                initial_backoff_sec * (backoff_multiplier**attempt)
                            ) + random.uniform(0.1, 0.4)
                            logger.warning(
                                f"Detected 500 error, retrying attempt {attempt + 1}: {unhandled_err}"
                            )
                            time.sleep(delay)
                            continue
                        return ToolExecutionResult(
                            status="ERROR_UPSTREAM_SERVICE",
                            http_code=500,
                            error=f"HTTP 500 Upstream Service Error: {unhandled_err}",
                            retryable=True,
                        ).model_dump()

                    logger.error(
                        f"Unhandled tool error in {func.__name__}: {unhandled_err}"
                    )
                    return ToolExecutionResult(
                        status="ERROR_INTERNAL",
                        error=f"Tool execution exception: {unhandled_err}",
                        retryable=False,
                    ).model_dump()

            return ToolExecutionResult(
                status="ERROR_RETRIES_EXHAUSTED",
                error=f"Exhausted {max_retries} attempts: {last_exception}",
                retryable=True,
            ).model_dump()

        return async_wrapper if is_async else sync_wrapper

    return decorator
