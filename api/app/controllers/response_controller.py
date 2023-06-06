import connexion
import six

from swagger_server.models.analysis import Analysis  # noqa: E501
from swagger_server import util


def response_time_post(video, x_api_key):  # noqa: E501
    """response_time_post

    Post a video # noqa: E501

    :param video: 
    :type video: strstr
    :param x_api_key: API Key
    :type x_api_key: str

    :rtype: Analysis
    """
    return 'do some magic!'

def response_time_get():
    """
    response_time_get
    Get response time without providing a video or API key

    :rtype: Analysis
    """
    return 'do some magic for GET!'
