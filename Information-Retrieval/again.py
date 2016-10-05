# -*- coding: utf-8 -*- 
import json,re,codecs
from collections import OrderedDict
import string

from stop_words import get_stop_words
		
#fname="search_tweets_ko.json"
fname="./14stream_1621.json"
twt=[]
sfname="./formatted_json/sports/14stream_1621.json"
rt_prefix="ttext_"
topic_dic=OrderedDict([('iphone7','Tech'),('us presidential election','Politics'),('democrat party','Politics'),('syrian war','World News'),\
('isis','World News'),('ISIS','World News'),('game of thrones','Entertainment'),('aryastark','Entertainment'),('jon snow','Entertainment'),('us open tennis','Sports'),('game-of-thrones','Entertainment'),('tennis','Sports'),('us open','Sports'),\
('syria','World News'),('war','World News'),('trump','Politics'),('clinton','Politics'),('daenerys','Entertainment'),('jamielannister','Entertainment'),('tyrianlannister','Entertainment'),\
('democratparty','Politics'),('Hilary','Politics'),('donaldtrump','Politics'),('uselections','Politics'),('2016election','Politics'),\
('tennis','Sports'),('us open','Sports'),('usmuslim','Politics'),('stan wawrinka','Sports'),('wawrinka','Sports'),('Novak','Sports'),('djokovic','Sports'),\
('Suriye','World News'),('sava','World News'),('iOS','Tech'),('iOS10','Tech'),('ios10','Tech'),('usopenfinal','Sports'),('usopen','Sports'),('usopen2016','Sports'),('djokovicwawrinka','Sports'),('usopenxespn','Sports'),('stanwawrinka','Sports'),('stan wawrinka','Sports'),\
('stanimal','Sports'),('trophy','Sports'),('wii','Sports'),('stantheman','Sports'),('api','Sports'),('tennis','Sports')])
'''
topic_dic=OrderedDict([('iphone7','Tech'),('us presidential election','Politics'),('democrat party','Politics'),('syrian war','World News'),\
('isis','World News'),('game of thrones','Entertainment'),('jon snow','Entertainment'),('us open tennis','Sports'),('game-of-thrones','Entertainment'),('tennis','Sports'),('us open','Sports'),\
('syria','World News'),('war','World News'),('trump','Politics'),('clinton','Politics')])
'''
topic_keys=topic_dic.keys()
punc=['@','#','.','?','!',',']
tr=[]
tr.append(punc)
stop_words=[]

def trunc_stop(tweet_text,tweet_lang):
	ttext_xx=tweet_text
	if(tweet_lang=='ko'):
		pass
	else:
		stop_words=get_stop_words(tweet_lang)
		
		tokens=ttext_xx.split()
		ttext_xx = ' '.join([word for word in ttext_xx.split() if word not in (get_stop_words(tweet_lang))])
		ttext_xx=re.sub(r"http\S+","",ttext_xx)
	
	return ttext_xx
	

def trunc_twt(tweet_text,tweet_lang,tr):
	ttext_xx=tweet_text
	for i in tr:
		#if(tweet_lang=='en'):
		try:
			if(ttext_xx.find(str(i))>-1):
				ttext_xx=string.replace(ttext_xx,str(i),'')
		except UnicodeEncodeError:
			print "Unicode error skipping"
			continue
	'''
			k=[]
			k.append(i)
			s=''.join(h for h in k)
			l=[x.encode('utf-8') for x in s]
			if(ttext_xx.find(l)>-1):
				ttext_xx=string.replace(ttext_xx,l,'')
			#print i
		else:
			if(ttext_xx.find(str(i)) > -1):
				ttext_xx=string.replace(ttext_xx,str(i),'')
			#print str(i)
	'''
	for i in punc:
		if(ttext_xx.find(str(i)) > -1):
			ttext_xx=string.replace(ttext_xx,str(i),'')
			
	
	return ttext_xx
			
import io
fout=io.open(sfname,'w',encoding='utf-8')
with io.open(fname,encoding='utf-8') as f:
	for line in f:
		#j={}
		j=json.loads(line)
		#j=dict(json.loads(line))
		twt.append(json.loads(line))
		res=[]
		
		tweet_date=j['created_at']
		d=tweet_date.split()
		t=d[3].split(':')
		mm=int(t[1])
		hr=int(t[0])
		tmp_ts=t[0]+":"+t[1]+":"+t[2]
		if(mm>=30):
			hr=(hr+1)%24
		rt=str(hr)+":00:00"
		tweet_date=string.replace(tweet_date,tmp_ts,rt)
		
		id_=j['id']
		tweet_text=j['text']
		tweet_lang=j['lang']
		ttext_en=""
		ttext_es=""
		ttext_ko=""
		ttext_tr=""
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
		
				
		tweet_urls=j['entities']['urls']
		tr.append(tweet_urls)
		tweet_emoticons=[]
		tr.append(tweet_emoticons)
		trunc_tweet=trunc_twt(tweet_text,tweet_lang,tr)
		trunc_tweet=trunc_stop(trunc_tweet,tweet_lang)
		
		'''
		tok=tweet_text.encode('utf-8').split('\u')
		for i in tok:
			if(i==''):
				pass
			else:
				emoticons="\u"+str(i[0:4])
				tweet_emoticons.append(emoticons)
		#emoticons = re.findall(ru'[\u0001-\uffff]', tweet_text)
		
		print tweet_emoticons
		'''
		#j['emojis']
		tweet_loc=j['geo']
		if(tweet_lang=='en'):
			ttext_en=trunc_tweet
		elif(tweet_lang=='es'):
			ttext_es=trunc_tweet
		elif(tweet_lang=='tr'):
			ttext_tr=trunc_tweet
		else:
			ttext_ko=trunc_tweet
		
		my_dic=OrderedDict([('tweet_date',tweet_date),('topic',top),('tweet_text',tweet_text),('tweet_lang',tweet_lang),('ttext_en',ttext_en),('ttext_es',ttext_es),('ttext_tr',ttext_tr),('ttext_ko',ttext_ko
		),('hashtags',hashtags),('mentions',mentions),('tweet_urls',tweet_urls),('tweet_emoticons',tweet_emoticons),('tweet_loc',tweet_loc)])
		
		res.append(my_dic)
		
		my=json.dumps(my_dic,ensure_ascii=False,encoding="utf-8")
		print my
		#json.dump(unicode(res),fout)
		#my_str=json.dump(res,ensure_ascii=False)
		#my_json_str=json.dump(res,ensure_ascii=False)
		#if isinstance(my_str,str):
			#my_str=my_str.decode("utf-8")
		
		#fout.write(unicode(json.dump(res,ensure_ascii=False)))
		#fout.write(my_str)
		fout.write(my)
		fout.write("\n".decode("utf-8"))

f.close()
fout.close()

'''
● topic : One of the five topics  
● tweet_text : Default field 
● tweet_lang : Language of the tweet from Twitter as a two letter code. 
● ttext_xx : For language specific fields where xx is at least one amongst en 
(English), es (Spanish), tr (Turkish) and ko (Korean) 
● hashtags, mentions, tweet_urls and tweet_emoticons for the respective self 
explanatory values 
● tweet_date : Date of the tweet, rounded to the nearest hour and in GMT 
● tweet_loc : Geolocation of the tweet.   
'''
