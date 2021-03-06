import json

import pytest

from mnist-function import app


@pytest.fixture()
def apigw_event():
    """ Generates API GW Event"""

    return {
    "body": "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",
    "resource": "/hello",
    "path": "/hello",
    "httpMethod": "POST",
    "isBase64Encoded": true,
    "queryStringParameters": {
      "foo": "bar"
    },
    "pathParameters": {
      "proxy": "/path/to/resource"
    },
    "stageVariables": {
      "baz": "qux"
    },
    "headers": {
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
      "Accept-Encoding": "gzip, deflate, sdch",
      "Accept-Language": "en-US,en;q=0.8",
      "Cache-Control": "max-age=0",
      "CloudFront-Forwarded-Proto": "https",
      "CloudFront-Is-Desktop-Viewer": "true",
      "CloudFront-Is-Mobile-Viewer": "false",
      "CloudFront-Is-SmartTV-Viewer": "false",
      "CloudFront-Is-Tablet-Viewer": "false",
      "CloudFront-Viewer-Country": "US",
      "Host": "1234567890.execute-api.us-east-1.amazonaws.com",
      "Upgrade-Insecure-Requests": "1",
      "User-Agent": "Custom User Agent String",
      "Via": "1.1 08f323deadbeefa7af34d5feb414ce27.cloudfront.net (CloudFront)",
      "X-Amz-Cf-Id": "cDehVQoZnx43VYQb9j2-nvCh-9z396Uhbp027Y2JvkCPNLmGJHqlaA==",
      "X-Forwarded-For": "127.0.0.1, 127.0.0.2",
      "X-Forwarded-Port": "443",
      "X-Forwarded-Proto": "https"
    },
    "requestContext": {
      "accountId": "123456789012",
      "resourceId": "123456",
      "stage": "prod",
      "requestId": "c6af9ac6-7b61-11e6-9a41-93e8deadbeef",
      "requestTime": "09/Apr/2015:12:34:56 +0000",
      "requestTimeEpoch": 1428582896000,
      "identity": {
        "cognitoIdentityPoolId": null,
        "accountId": null,
        "cognitoIdentityId": null,
        "caller": null,
        "accessKey": null,
        "sourceIp": "127.0.0.1",
        "cognitoAuthenticationType": null,
        "cognitoAuthenticationProvider": null,
        "userArn": null,
        "userAgent": "Custom User Agent String",
        "user": null
      },
      "path": "/prod/hello",
      "resourcePath": "/hello",
      "httpMethod": "POST",
      "apiId": "1234567890",
      "protocol": "HTTP/1.1"
    }
  }
  


def test_lambda_handler(apigw_event, mocker):

    ret = app.lambda_handler(apigw_event, "")
    data = json.loads(ret["body"])

    assert ret["statusCode"] == 200
    assert "inference" in ret["body"]
    assert data["inference"] == "1"
