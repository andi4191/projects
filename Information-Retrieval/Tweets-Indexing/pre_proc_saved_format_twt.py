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

		
fname="../test_dir/search_and_saved_11_sept_2202tr.json"
#fname="processed_json/*.json"
#fname="11stream_0124hr.json"
twt=[]
sfname="../formatted_searched_jsons/search_and_saved_11_sept_2202tr.json"
rt_prefix="ttext_"

topic_dic=OrderedDict([('iphone7','Tech'),('us presidential election','Politics'),('democrat party','Politics'),('syrian war','World News'),\
('isis','World News'),('game of thrones','Entertainment'),('jon snow','Entertainment'),('us open tennis','Sports'),('game-of-thrones','Entertainment'),('tennis','Sports'),('us open','Sports'),\
('syria','World News'),('war','World News'),('trump','Politics'),('clinton','Politics'),('daenerys','Entertainment'),('jamielannister','Entertainment'),('tyrianlannister','Entertainment')])
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
			
import io
fout=io.open(sfname,'w',encoding='utf-8')
#for fname in filname:
with io.open(fname,encoding='utf-8') as f:
	for line in f:
		#j={}
		j=json.loads(line)
		#j=dict(json.loads(line))
		twt.append(json.loads(line))
		res=[]
		#print type(j)
		tweet_date=j['tweet_date']
		d=tweet_date.split()
		#print d
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
		
		#id_=j['id']
		tweet_text=j['tweet_text']
		tweet_lang=j['tweet_lang']
		text_en=""
		text_es=""
		text_ko=""
		text_tr=""
		ttext_xx=rt_prefix+str(tweet_lang)
		ht=j['hashtags']
		hashtags=ht
		#for i in ht:
		#	hashtags.append(i)
		#	tr.append(i['text'])
		#	print hashtags
		mt=j['mentions']
		mentions=mt
		#for k in mt:
		#	mentions.append(k['screen_name'])
		#	tr.append(k['screen_name'])		
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
		
		#[0]['url']
		#tr.append(tweet_urls)
		tweet_emoticons=[]
		tweet_emoticons=locate_emoji(tweet_text)
		pat=re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
		tweet_emoticons.append(re.findall(pat,tweet_text))
		
		tr.append(tweet_emoticons)
		trunc_tweet=trunc_twt(tweet_text,tweet_lang,hashtags,mentions)
		#trunc_tweet=re.sub(r'[:)(]*','',trunc_tweet)
		
		
		trunc_tweet=re.sub(pat,'',trunc_tweet)
		#emo=detect_emotics(trunc_tweet)
		#tweet_emoticons.append(emo)
		#trunc_tweet=trunc_stop(trunc_tweet,tweet_lang)
		
		tweet_loc=j['tweet_loc']
		#if(j['tweet_loc']):
		#	tweet_loc=j['coordinates']['coordinates']
		if(tweet_lang=='en'):
			text_en=trunc_tweet
		elif(tweet_lang=='es'):
			text_es=trunc_tweet
		elif(tweet_lang=='tr'):
			text_tr=trunc_tweet
		else:
			text_ko=trunc_tweet
		
		my_dic=OrderedDict([('tweet_date',tweet_date),('topic',top),('tweet_text',tweet_text),('tweet_lang',tweet_lang),('text_en',text_en),('text_es',text_es),('text_tr',text_tr),('text_ko',text_ko
		),('hashtags',hashtags),('mentions',mentions),('tweet_urls',tweet_urls),('tweet_emoticons',tweet_emoticons),('tweet_loc',tweet_loc)])
		
		res.append(my_dic)
		
		my=json.dumps(my_dic,ensure_ascii=False,encoding="utf-8")
		print my
		
		fout.write(my)
		fout.write("\n".decode("utf-8"))

f.close()
fout.close()
