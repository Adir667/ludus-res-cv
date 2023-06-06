# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.analysis import Analysis  # noqa: E501
from swagger_server.test import BaseTestCase


class TestResponseController(BaseTestCase):
    """ResponseController integration test stubs"""

    def test_response_time_post(self):
        """Test case for response_time_post

        
        """
        headers = [('x_api_key', 'x_api_key_example')]
        data = dict(video='video_example')
        response = self.client.open(
            '/Adir667/Ludus-res/3.0.0/response_time',
            method='POST',
            headers=headers,
            data=data,
            content_type='multipart/form-data')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
