from sentimetal_mod import sentiment
# print(sentiment("This movie was awesome, the acting was great and there are a lot of fun. Oh, yea!"))
# print(sentiment("This movie was junk. It sucks. Horrible movie and it is 0 out of 10."))
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from multiprocessing import Process
import time
from http.client import IncompleteRead

#consumer key, consumer secret, access token, access secret.
ckey="lEgsw7Z5NrkGwZGbpV4kk3FtJ"
csecret="DnnMewV8KIJictgsV54WZHa98kT1mFpRgh5dma5IIII54iLJXb"
atoken="817588581208363009-7qxiPioKXTJbbMu03Pwzf34r1J960Dv"
asecret="9YZaUych0nMB3PMhZxHMO75inO5cXEjJoDJj68vM04lLo"

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
style.use("ggplot")

class listener(StreamListener):
	    def on_data(self, data):
	        time.sleep(0.2)
	        all_data = json.loads(data)
	        if "text" in all_data:
	        	tweet = all_data["text"]
		        sentiment_value , confidence = sentiment(tweet)
		        #print(tweet, sentiment_value, confidence)
		        if confidence >= 0.7:
		        	with open("tweeter_out.txt", "a") as f:
		        		print(sentiment_value, file=f, end='\n')
	        return True

	    def on_error(self, status):
	        print(status)

def get_data_from_twitter():
	with open("tweeter_out.txt", "w") as f:
		pass

	auth = OAuthHandler(ckey, csecret)
	auth.set_access_token(atoken, asecret)

	while 1:
		try:
			twitterStream = Stream(auth, listener())
			twitterStream.filter(track=["happy", "excellent", "great"], languages=['en'])
		except IncompleteRead:
			continue
		except KeyboardInterrupt:
			twitterStream.disconnect()
			break

def animate(i):
	with open("tweeter_out.txt", "r") as f:
		pull_datas = f.read().split("\n")
		
	xar = []
	yar = []
	x = 0
	y = 0

	for l in pull_datas:
		x += 1
		if "pos" in l:
			y += 1
		else:
			y -= 0.8
		xar.append(x)
		yar.append(y)
	
	ax1.clear()
	ax1.plot(xar, yar)


def live_data_visualization():
	ani = animation.FuncAnimation(fig, animate, interval=3000)
	plt.show()

def main():
	p1 = Process(target=get_data_from_twitter)
	p2 = Process(target=live_data_visualization)

	p1.start()
	p2.start()

	p1.join()
	p2.join()

if __name__ == '__main__':
	main()