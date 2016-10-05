# -*- coding: utf-8 -*- 
import json,re,codecs
from collections import OrderedDict
import string
from stop_words import get_stop_words
import os

filname=[]
for files in os.listdir("./"):
	if(files.endswith(".json")):
		filname.append(files)

		
#fname="../mom_account/14stream_sports_politics_1433.json"
fname="../final_tweets/search_and_saved_14_sept_2216tr.json"
sfname="../final_tweets/rev_loc/search_and_saved_14_sept_2216tr.json"
#fname="processed_json/*.json"
#fname="11stream_0124hr.json"
twt=[]

rt_prefix="ttext_"

topic_dic=OrderedDict([('Airpod','Tech'),('iphone7','Tech'),('us presidential election','Politics'),('democrat party','Politics'),('syrian war','World News'),\
('isis','World News'),('ISIS','World News'),('game of thrones','Entertainment'),('aryastark','Entertainment'),('jon snow','Entertainment'),('us open tennis','Sports'),('game-of-thrones','Entertainment'),('tennis','Sports'),('us open','Sports'),\
('syria','World News'),('war','World News'),('trump','Politics'),('clinton','Politics'),('daenerys','Entertainment'),('jamielannister','Entertainment'),('tyrianlannister','Entertainment'),\
('democratparty','Politics'),('Hilary','Politics'),('donaldtrump','Politics'),('uselections','Politics'),('2016election','Politics'),\
('tennis','Sports'),('us open','Sports'),('usmuslim','Politics'),('stan wawrinka','Sports'),('wawrinka','Sports'),('Novak','Sports'),('djokovic','Sports'),\
('Suriye','World News'),('sava','World News'),('iOS','Tech'),('iOS10','Tech'),('ios10','Tech'),('usopenfinal','Sports'),('usopen','Sports'),('usopen2016','Sports'),('djokovicwawrinka','Sports'),('usopenxespn','Sports'),('#stanwawrinka','Sports'),('stan wawrinka','Sports'),\
('stanimal','Sports'),('trophy','Sports'),('wii','Sports'),('stantheman','Sports'),(' api ','Sports'),('IPHONE','Tech'),('iPhone','Tech'),('tenes','Sports'),('android','Tech'),('Apple','Tech'),('iPad','Tech'),('el qaide','World News'),('isaguerrero','Sports'),('tenias','Sports')])
'''
topic_dic=OrderedDict([('us presidential election','Politics'),('democrat party','Politics'),('democratparty','Politics'),\
('Hilary','Politics'),('trump','Politics'),('donaldtrump','Politics'),('uselections','Politics'),('2016election','Politics'),('tennis','Sports'),('us open','Sports'),\
('usmuslim','Politics'),('stan wawrinka','Sports'),('wawrinka','Sports'),('Novak','Sports'),('djokovic','Sports'),('clinton','Politics')])
'''
topic_keys=topic_dic.keys()
punc=['@','#','.','?','!',',']
tr=[]
tr.append(punc)
stop_words=[]

def map_month(st):
	dic={'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
	return str(dic[st])

def trunc_stop(tweet_text,tweet_lang):
	ttext_xx=tweet_text
	if(tweet_lang=='ko'):
		pass
	else:
		stop_words=get_stop_words(tweet_lang)
		
		tokens=ttext_xx.split()
		ttext_xx = ' '.join([word for word in ttext_xx.split() if word not in (get_stop_words(tweet_lang))])
		#ttext_xx=re.sub(r"http\S+","",ttext_xx)
	
	return ttext_xx


def locate_emoji(tweet_text):
	emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" u"\U0001F466-\U0001F9FF" u"\u2764-\u2764" "]+", flags=re.UNICODE)
	emot=re.findall(emoji_pattern,tweet_text)
	return emot

def detect_emotics(tweet_text):
	ttext=[]
	emots=[':-)',':)',':(',':-(',':o\)', ':c\)', ':>', '=\]', '=\)', ':^\)',':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3','B^D']
	for i in emots:
		if(tweet_text.find(str(i))>-1):
			print i
			ttext.append(str(i))
			#tweet_text=re.sub(i,'',tweet_text)
	return ttext

def trunc_emotics(tweet_text):
	ttext=[]
	emots=[':-)',':)',':(',':-(',':o\)', ':c\)', ':>', '=\]', '=\)', ':^\)',':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3','B^D']
	for i in emots:
		if(tweet_text.find(str(i))>-1):
			print i
			ttext.append(i)
			tweet_text=re.sub(str(i),'',tweet_text)
	
	return tweet_text
	
          


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
	emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" u"\U0001F466-\U0001F9FF" u"\u2764-\u2764" "]+", flags=re.UNICODE)
	#emot=re.findall(emoji_pattern,ttext_xx)
	ttext_xx= emoji_pattern.sub(r'', ttext_xx) # no emoji
	#ttext_xx=trunc_emotics(ttext_xx)
	
	for i in punc:
		if(ttext_xx.find(str(i)) > -1):
			ttext_xx=string.replace(ttext_xx,str(i),'')
			
	
	return ttext_xx
count=0			
import io
fout=io.open(sfname,'w',encoding='utf-8')
#for fname in filname:
with io.open(fname,encoding='utf-8') as f:
	for line in f:
		count+=1
		#j={}
		j=json.loads(line)
		#j=dict(json.loads(line))
		twt.append(json.loads(line))
		res=[]
		#print type(j)
		tweet_date=j['tweet_date']
		tweet_text=j['tweet_text']
		tweet_lang=j['tweet_lang']
		text_en=j['text_en']
		text_es=j['text_es']
		text_ko=j['text_ko']
		text_tr=j['text_tr']
		hashtags=j['hashtags']
		mentions=j['mentions']
		top=j['topic']
		tweet_urls=j['tweet_urls']
		tweet_emoticons=j['tweet_emoticons']
		tweet_loc=""
		val=[]
		val=j['tweet_loc']
		if(val):
			print count
			val=val[::-1]
			tweet_loc=','.join(str(x) for x in val)
			
			
			
		
		my_dic=OrderedDict([('tweet_date',tweet_date),('topic',top),('tweet_text',tweet_text),('tweet_lang',tweet_lang),('text_en',text_en),('text_es',text_es),('text_tr',text_tr),('text_ko',text_ko
		),('hashtags',hashtags),('mentions',mentions),('tweet_urls',tweet_urls),('tweet_emoticons',tweet_emoticons),('tweet_loc',str(tweet_loc))])
		
		res.append(my_dic)
		
		my=json.dumps(my_dic,ensure_ascii=False,encoding="utf-8")
		print my
		
		fout.write(my)
		fout.write("\n".decode("utf-8"))

f.close()
fout.close()
