# -*- coding: utf-8 -*- 


### code to save tweets in json###
import sys
import tweepy
import json
import codecs

consumer_key=""
consumer_secret=""
access_key=""
access_secret=""


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)
f = codecs.open('19stream_korean_tweets_1540.json', 'a',encoding='utf-8')

class CustomStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print status.text

    def on_data(self, data):
		
		json_data = json.loads(data)
		try:
			#print json_data['created_at']
			#emojis=
			txt=json_data['text']
			if(txt[0:2]=="RT"):
				pass
			else:
				#json.dump(json_data,f,encoding='utf-8')
				json.dump(json_data,f,ensure_ascii=False)
				#For easy readability
				print "Dumping stream data..."
				f.write("\n".decode('utf-8'))
		except KeyError:
				print "Key Error"
				pass

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True # Don't kill the stream

sapi = tweepy.streaming.Stream(auth, CustomStreamListener())
sapi.filter(track=['iOS','watch2','apple2016','Apple','iphone','applephone','airpod','ipod','iphone7','ios9','ipad','usopen','usopen2016','djokovicwawrinka','usopenxespn','stanwawrinka','stan wawrinka','atp','USOpen','andyroddick','Clijsterskim','Novak','Nadal','ATP World Tour'],languages=['ko'])
f.close()
