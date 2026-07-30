# Copyright (c) 2025, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import (
    Any,
    Dict,
    Optional,
)

from datacustomcode.named_credential.base import NamedCredential
from datacustomcode.named_credential.types.http_request import HTTPRequest
from datacustomcode.named_credential.types.http_response import HTTPResponse
from datacustomcode.named_credential.types.http_response_builder import (
    HTTPResponseBuilder,
)


class DefaultNamedCredential(NamedCredential):
    """
    Executes the callout directly via :class:`DirectCalloutTransport`, resolving
    the URL from the Named Credential Connect API (falling back to
    ``credential.json``) and injecting auth from ``credential.json``.
    """

    CONFIG_NAME = "DefaultNamedCredential"

    def __init__(
        self,
        credentials_profile: str = "default",
        sf_cli_org: Optional[str] = None,
        credential_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._credentials_profile = credentials_profile
        self._sf_cli_org = sf_cli_org
        self._credential_file = credential_file
        self._transport: Optional[Any] = None

    def request(
        self,
        request: HTTPRequest,
        body: Optional[Dict[str, Any]] = None,
    ) -> HTTPResponse:
        callout_response = self.callout_json(
            request, json.dumps(body) if body is not None else ""
        )

        raw_body = callout_response.get("body") or ""
        data: Optional[Any] = None
        if raw_body:
            try:
                # Preserve any JSON value: object, array, or scalar.
                data = json.loads(raw_body)
            except json.JSONDecodeError:
                data = None

        response_dict = {
            "status_code": callout_response.get("http_status_code"),
            "headers": callout_response.get("headers", {}),
            "data": data,
        }
        return HTTPResponseBuilder.build(response_dict)

    def callout_json(
        self,
        request: HTTPRequest,
        body: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Raw string-in/string-out callout returning the unparsed response.

        Unlike :meth:`request`, the ``body`` is sent verbatim (never re-parsed)
        and the response is returned as a raw ``{http_status_code, headers,
        body}`` dict rather than a parsed :class:`HTTPResponse`.
        """
        # Callout request shape sent to the transport.
        callout_request = {
            "path": request.url,
            "method": request.method,
            "headers": dict(request.headers),
            "body": body if body is not None else "",
        }
        callout_response = self._callout(callout_request)
        return {
            "http_status_code": callout_response.get("http_status_code"),
            "headers": callout_response.get("headers", {}),
            "body": callout_response.get("body") or "",
        }

    def _callout(self, callout_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the callout via the transport.

        Returns a dict with ``http_status_code``, ``headers``, and ``body``.
        """
        result: Dict[str, Any] = self._get_transport().callout(callout_request)
        return result

    def _get_transport(self) -> Any:
        if self._transport is None:
            from datacustomcode.named_credential.direct.transport import (
                DirectCalloutTransport,
            )

            self._transport = DirectCalloutTransport(
                credentials_profile=self._credentials_profile,
                sf_cli_org=self._sf_cli_org,
                credential_file=self._credential_file,
            )
        return self._transport
