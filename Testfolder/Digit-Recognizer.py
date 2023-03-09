import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json


class TweetsListener(StreamListener):
  """
  A listener class to stream real-time tweets based on a keyword or hashtag.
  """

  def __init__(self, csocket):
      """
      Constructor for the TweetsListener class.

      Parameters:
      csocket (socket): The client socket object.
      """
      self.client_socket = csocket

  def on_data(self, data):
      """
      This method gets called when new data is received from the stream.

      Parameters:
      data (str): A JSON string representing the tweet data.

      Returns:
      bool: True if the function executes successfully, False otherwise.
      """
      try:
          msg = json.loads( data )
          print( msg['text'].encode('utf-8') )
          self.client_socket.send( msg['text'].encode('utf-8') )
          return True
      except BaseException as e:
          print("Error on_data: %s" % str(e))
      return True

  def on_error(self, status):
      """
      This method gets called when an error occurs while streaming data.

      Parameters:
      status (int): The HTTP error code returned by the server.

      Returns:
      bool: True if the function executes successfully, False otherwise.
      """
      print(status)
      return True


def sendData(c_socket):
  """
  A function to set up the authentication credentials and filter the tweets based on a keyword/hashtag.

  Parameters:
  c_socket (socket): The client socket object.

  Returns:
  None
  """
  auth = OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_secret)

  twitter_stream = Stream(auth, TweetsListener(c_socket))
  twitter_stream.filter(track=['soccer'])


if __name__ == "__main__":
  # Set up a socket object
  s = socket.socket()
  host = "127.0.0.1"
  port = 5555
  s.bind((host, port))
  print("Listening on port: %s" % str(port))
  s.listen(5)
  
  # Wait for a client connection to establish
  c, addr = s.accept()
  print( "Received request from: " + str( addr ) )

  # Stream the tweets and send them to the client
  sendData( c )