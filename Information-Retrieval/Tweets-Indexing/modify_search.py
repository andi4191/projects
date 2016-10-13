# -*- coding: utf-8 -*- 
import tweepy
import json, re
import codecs
import time,string
from stop_words import get_stop_words
from collections import OrderedDict

consumer_key=""
consumer_secret=""
access_key=""
access_secret=""

auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_key,access_secret)

api=tweepy.API(auth)
#lang=['en', 'tr', 'ko', 'es']
lang=['en','es','tr','ko']
#topics=['iphone7','us presidential election','democrat party','syrian war','isis','game of thrones','jon snow','us open tennis','trump','clinton','apple watch','applewatch','apple-watch','watch2']

rt_prefix="ttext_"

topic_dic=OrderedDict([('tennis','Sports'),('iphone7','Tech'),('us presidential election','Politics'),('democrat party','Politics'),('syrian war','World News'),\
('isis','World News'),('game of thrones','Entertainment'),('jon snow','Entertainment'),('us open tennis','Sports'),('game-of-thrones','Entertainment'),('tennis','Sports'),('us open','Sports'),\
('syria','World News'),('war','World News'),('trump','Politics'),('clinton','Politics'),('timcook','Tech'),('tim cook','Tech'),('iphone 7','Tech'),('daenerys','Entertainment'),('jamielannister','Entertainment'),('tyrianlannister','Entertainment')])
topic_keys=topic_dic.keys()
punc=['@','#','.','?','!',',']
tr=[]
tr.append(punc)
stop_words=[]


file_name="search_and_saved_14_sept_2216"
#emoji_pattern = re.compile('[\u001F300-\u001F64F]')

def map_month(st):
	dic={'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
	return str(dic[st])

def locate_emoji(tweet_text):
	emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" u"\U0001F466-\U0001F9FF" "]+", flags=re.UNICODE)
	emot=re.findall(emoji_pattern,tweet_text)
	return emot


def trunc_twt(tweet_text,tweet_lang,hashtags,mentions):
	
	ttext_xx=tweet_text
	for i in hashtags:
		ttext_xx=re.sub(i,"",ttext_xx)
		#print i
	for i in mentions:
		ttext_xx=re.sub(i,"",ttext_xx)
		#print i
	ttext_xx=re.sub(r"http\S+","",ttext_xx)
	#for i in emoticons:
	#	ttext_xx=re.sub(i,"",ttext_xx)
	emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" u"\U0001F466-\U0001F9FF" "]+", flags=re.UNICODE)
	#emot=re.findall(emoji_pattern,ttext_xx)
	ttext_xx= emoji_pattern.sub(r'', ttext_xx) # no emoji
	
	for i in punc:
		if(ttext_xx.find(str(i)) > -1):
			ttext_xx=string.replace(ttext_xx,str(i),'')
			
	
	return ttext_xx
			


def trunc_stop(tweet_text,tweet_lang):
	
	ttext_xx=tweet_text
	print tweet_text
	if(tweet_lang=='ko'):
		pass
	else:
		stop_words=get_stop_words(tweet_lang)
		
		tokens=ttext_xx.split()
		ttext_xx = ' '.join([word for word in ttext_xx.split() if word not in (get_stop_words(tweet_lang))])
		#ttext_xx=re.sub(r"http\S+","",ttext_xx)
	
	return ttext_xx

#def trunc_twt(tweet_text,tweet_lang,tr,ht,mentions,emoticons):

#def trunc_twt(tweet_text,tweet_lang,tr):
'''
def trunc_twt(tweet_text,tweet_lang,tr,ht,mentions,emoticons):
	ttext_xx=tweet_text
	print tweet_text
	
	#for i in ht:
	#	print ttext_xx + "andi"
	#	#string.replace(ttext_xx,str(i),"")
	#	print ttext_xx
	#print ttext_xx
	#for i in mentions:
	#	re.sub(i,"",ttext_xx)
	#print ttext_xx
	#for i in emoticons:
	#	re.sub(i,"",ttext_xx)
	#print ttext_xx
'''
'''
	for i in tr:
		#if(tweet_lang=='en'):
		try:
			if(ttext_xx.find(str(i))>-1):
				ttext_xx=string.replace(ttext_xx,str(i),'')
		except UnicodeEncodeError:
			print "Unicode error skipping"
			continue
'''
'''
	ttext_xx=re.sub(r"@\S+","",ttext_xx)
	ttext_xx=re.sub(r"#\S+","",ttext_xx)
	ttext_xx=re.sub(r"http\S+","",ttext_xx)
	#ttext_xx=re.sub(r'?','',ttext_xx)
	#ttext_xx=re.sub(r"!","",ttext_xx)
	#ttext_xx=re.sub(r",","",ttext_xx)
	#ttext_xx=re.sub(r".","",ttext_xx)
	return ttext_xx
'''
import codecs
while 1:
	for i in lang:
		f=codecs.open(file_name+str(i)+".json",'a',encoding='utf-8')
		for k in topic_keys:
			topic=k
			query=str(k)
			
			#api = tweepy.API(auth)
			max_tweets=100
			
			for status in tweepy.Cursor(api.search,  q=query,lang=i,since="2015-09-01").items(max_tweets):
				#dic={}
				#dic=unicode(status._json)
				#json.dump(dic,f,encoding="utf-8",ensure_ascii=False)
				#f.write("\n".decode('utf-8'))
				res=[]
				#twt.append(json.loads(line))
				j=status._json
				tweet_text=j['text']
				if(tweet_text[0:2]=="RT"):
					continue
				tweet_date=j['created_at']
				d=tweet_date.split()
				month=map_month(d[1])
				day=d[2]
				year=d[5]
				zone="Z"
				t=d[3].split(':')
				mm=int(t[1])
				hr=int(t[0])
				tmp_ts=t[0]+":"+t[1]+":"+t[2]
				if(mm>=30):
					hr=(hr+1)%24
		
				hr=str(hr).rjust(2, '0')
				#print hr
				rt=str(hr)+":00:00"
				final_dat=[year,'-',month,'-',day,'T',rt,zone]
				#print final_dat
				tweet_date=''.join([k for k in final_dat])
		
				id_=j['id']
				
				tweet_lang=j['lang']
				text_en=""
				text_es=""
				text_ko=""
				text_tr=""
				ttext_xx=rt_prefix+str(tweet_lang)
				ht=j['entities']['hashtags']
				hashtags=[]
				for i in ht:
					hashtags.append(i['text'])
					tr.append(i['text'])
				#print hashtags
				mt=j['entities']['user_mentions']
				mentions=[]
				for k in mt:
					mentions.append(k['screen_name'])
					tr.append(k['screen_name'])		
				top=""
				for i in topic_keys:
					if(tweet_text.lower().strip().find(i.lower().strip()) > -1):
						top=topic_dic[i]
						break
				#pat1=re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
				pat1=re.compile(r"http\S+")
				#tweet_urls=j['entities']['urls']
				tweet_urls=""
				tweet_urls=re.findall(pat1,tweet_text)							
				tr.append(tweet_urls)
				
				tweet_emoticons=[]
				tweet_emoticons=locate_emoji(tweet_text)
				pat=re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
				tweet_emoticons.append(re.findall(pat,tweet_text))
				
				trunc_tweet=trunc_twt(tweet_text,tweet_lang,hashtags,mentions)
				trunc_tweet=re.sub(pat,'',trunc_tweet)
				tweet_loc=[]
				if(j['coordinates']):
					tweet_loc=j['coordinates']['coordinates']
				if(tweet_lang=='en'):
					text_en=trunc_tweet
				elif(tweet_lang=='es'):
					text_es=trunc_tweet
				elif(tweet_lang=='tr'):
					text_tr=trunc_tweet
				else:
					text_ko=trunc_tweet
		
				my_dic=OrderedDict([('tweet_date',tweet_date),('topic',top),('tweet_text',tweet_text),('tweet_lang',tweet_lang),('text_en',text_en),('text_es',text_es),('text_tr',text_tr),('text_ko',text_ko
				),('hashtags',hashtags),('mentions',mentions),('tweet_urls',tweet_urls),('tweet_emoticons',tweet_emoticons),('tweet_loc',tweet_loc),('tweet_id',id_)])
		
				res.append(my_dic)
		
				my=json.dumps(my_dic,ensure_ascii=False,encoding="utf-8")
				time.sleep(10)
				#print my
				#json.dump(unicode(res),fout)
				#my_str=json.dump(res,ensure_ascii=False)
				#my_json_str=json.dump(res,ensure_ascii=False)
				#if isinstance(my_str,str):
				#my_str=my_str.decode("utf-8")
		
				#fout.write(unicode(json.dump(res,ensure_ascii=False)))
				#fout.write(my_str)
				f.write(my)
				f.write("\n".decode("utf-8"))
				
				
	
		f.close()
		

