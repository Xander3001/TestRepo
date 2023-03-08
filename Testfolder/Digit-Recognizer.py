import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json


class TweetsListener(StreamListener):
    """
    Streaming tweets listener class used for printing tweets and sending them to a socket.
    """
    def __init__(self, csocket):
        """
        Constructor method for the TweetsListener class.
        :param csocket: Socket object used for sending data.
        """
        self.client_socket = csocket

    def on_data(self, data):
        """
        Method called when new data is received from the stream.
        :param data: Raw data received from the stream.
        :return: True if data is processed correctly, otherwise True.
        """
        try:
            msg = json.loads(data)
            print(msg['text'].encode('utf-8'))
            self.client_socket.send(msg['text'].encode('utf-8'))
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        """
        Method called when there is an error in the stream.
        :param status: Error status message.
        :return: True to continue listening.
        """
        print(status)
        return True


def sendData(c_socket):
    """
    Method used to set up tweet streaming and send received tweets to a socket.
    :param c_socket: Socket object used for sending data.
    """
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    twitter_stream = Stream(auth, TweetsListener(c_socket))
    twitter_stream.filter(track=['soccer'])


if __name__ == "__main__":
    # Set up your credentials
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_secret = ''

    s = socket.socket()  # Create a socket object
    host = "127.0.0.1"  # Get local machine name
    port = 5555  # Reserve a port for your service.
    s.bind((host, port))  # Bind to the port

    print("Listening on port: %s" % str(port))

    s.listen(5)  # Now wait for client connection.
    c, addr = s.accept()  # Establish connection with client.

    print("Received request from: " + str(addr))

    sendData(c)