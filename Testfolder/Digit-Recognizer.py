import tweepy #import the tweepy library for accessing the Twitter API
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket  #import the socket library for networking
import json    #import the json library for data serialization


# Set up your credentials
consumer_key=''
consumer_secret=''
access_token =''
access_secret=''


class TweetsListener(StreamListener):

  def __init__(self, csocket):
      #The __init__ function is called when an object is created from this class.
      #The 'self' parameter refers to the object itself and 'csocket' is a parameter passed in when an object is created
      #Here, the client socket is initialized
      self.client_socket = csocket

  def on_data(self, data):
      #The on_data function is called every time a new tweet is received
      try:
          msg = json.loads( data )
          print( msg['text'].encode('utf-8') )
          self.client_socket.send( msg['text'].encode('utf-8') ) #encode the tweet in utf-8 and send it to the client socket
          return True
      except BaseException as e:
          print("Error on_data: %s" % str(e))
      return True

  def on_error(self, status):
      #The on_error function is called when an error occurs during data streaming
      print(status)
      return True

def sendData(c_socket):
  #The sendData function establishes a connection to the Twitter API using the credentials provided and filters tweets by keywords
  auth = OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_secret)

  #Create a Stream object called twitter_stream and filter tweets by the keyword 'soccer'
  twitter_stream = Stream(auth, TweetsListener(c_socket))
  twitter_stream.filter(track=['soccer'])

if __name__ == "__main__":
  s = socket.socket()         # Create a socket object
  host = "127.0.0.1"     # Get local machine name
  port = 5555                 # Reserve a port for your service.
  s.bind((host, port))        # Bind to the port

  print("Listening on port: %s" % str(port))

  s.listen(5)                 # Now wait for client connection.
  c, addr = s.accept()        # Establish connection with client.

  print( "Received request from: " + str( addr ) )

  sendData( c )  #call the sendData function with the client socket object as a parameter