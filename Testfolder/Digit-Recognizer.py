import tweepy            # Importing the tweepy library for accessing Twitter's API
from tweepy import OAuthHandler       # OAuthHandler module for authentication
from tweepy import Stream             # Stream module for transmitting data
from tweepy.streaming import StreamListener   # StreamListener module for receiving data
import socket                          # Importing socket module
import json                           

# Set up your credentials. Add your Twitter API credentials here.
consumer_key=''
consumer_secret=''
access_token =''
access_secret=''

class TweetsListener(StreamListener):
  
  # Initializing TweetsListener class with client socket
  def __init__(self, csocket):
      self.client_socket = csocket
  
  # Method to receive data
  def on_data(self, data):
      try:
          msg = json.loads( data )   # Loading data from json file
          
          # Printing text contained in the message and encoding it in utf-8 format
          print( msg['text'].encode('utf-8') ) 
          
          # Sending the encoded text message to the client socket
          self.client_socket.send( msg['text'].encode('utf-8') ) 
          return True
      except BaseException as e:
          print("Error on_data: %s" % str(e))
      return True
  
  # Method to handle errors
  def on_error(self, status):
      print(status)
      return True
  
# Function to send data to the client socket
def sendData(c_socket):
  auth = OAuthHandler(consumer_key, consumer_secret)   # Authenticating the credentials
  auth.set_access_token(access_token, access_secret)

  # Initializing the Stream object with auth and TweetsListener to receive data
  twitter_stream = Stream(auth, TweetsListener(c_socket))
  twitter_stream.filter(track=['soccer'])  # Filtering the stream data with 'soccer' keyword

if __name__ == "__main__":
  s = socket.socket()         # Create a socket object
  host = "127.0.0.1"     # Get local machine name
  port = 5555                 # Reserve a port for your service.
  s.bind((host, port))        # Bind to the port

  print("Listening on port: %s" % str(port))

  s.listen(5)                 # Now wait for client connection.
  c, addr = s.accept()        # Establish connection with client.

  # Printing the address of the client
  print( "Received request from: " + str( addr ) )

  sendData( c )    # Calling the sendData function to send the data to the client socket