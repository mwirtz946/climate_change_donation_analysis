import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import punkt
import nltk
from nltk import word_tokenize
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import string, re

import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

data = pd.read_csv('data/twitter_sentiment_data.csv')
class_labels = ['Anti Man-Made','Neutral','Man-Made','News']

def lemmatize_tweet(data):
    '''
    Function to lemmatize tweets

    Input
    -----
    data : str

    Optional Input
    --------------
    None

    Output
    ------
    String containing lemmatized tweets
    '''   
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    lem_data = tokenize_single(data,r'[a-z]+')
    lem_data = [lemmatizer.lemmatize(word) for word in lem_data if word not in stop_words]
    lem_data = [word for word in lem_data if len(word) > 2]
    lem_tweet = untokenize_single(lem_data)
    lem_tweet = lem_tweet.strip()
    
    return lem_tweet

def clean_tweet(data):
    '''
    Function to clean tweets

    Input
    -----
    data : str

    Optional Input
    --------------
    None

    Output
    ------
    Cleaned tweets as strings
    ''' 
    #removing hashtags, hyperlinks, mentions
    data = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",data).split())
    # removing mentions
    data = re.sub('(@[A-Za-z0-9]+)', '', data)
    # removing links
    data = re.sub(r'http\S+', '', data)
    data = re.sub(r'pic\.\S+', '', data)
    # convert contractions
    data = decontracted(data)
    # removing retweets
    data = re.sub("RT",'',data).strip()
    # making lowercase
    data = data.lower()
    
    # filtering for just letters
    data = tokenize_single(data, r'[a-zA-Z]+')
    data = untokenize_single(data)
    
    return data

def decontracted(phrase):
    '''
    Function to convert contractions

    Input
    -----
    data : str

    Optional Input
    --------------
    None

    Output
    ------
    String containing elements from input list
    
    Source
    ------
    https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    ''' 
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def untokenize_single(data):
    '''
    Function to untokenize a single list.

    Input
    -----
    data : list (str)

    Optional Input
    --------------
    None

    Output
    ------
    String containing elements from input list
    '''
    joined = ','.join(data)
    new_data = joined.replace(',',' ')
    return new_data

def tokenize_single(data, parameters):
    '''
    Function to tokenize any single string.
    
    Input
    -----
    data : str 
    parameters : Regex Filter
        Ex: r'[a-zA-Z]+'
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Tokenized data
    '''   
    tokenizer = RegexpTokenizer(parameters)
    data = tokenizer.tokenize(data)
    return data

def tokenize(data, parameters):
    '''
    Function to tokenize any series of strings.
    
    Input
    -----
    data : str 
    parameters : Regex Filter
        Ex: r'[a-zA-Z]+'
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Tokenized data
    '''
    tokenizer = RegexpTokenizer(parameters)
    data.message = data.message.apply(lambda x: tokenizer.tokenize(x))
    return data.message

def untokenize(data):
    '''
    Function to untokenize a series of lists.

    Input
    -----
    data : list (str)

    Optional Input
    --------------
    None

    Output
    ------
    String containing elements from input list
    '''
    data.message = data.message.apply(lambda x: ','.join(x))
    data.message = data.message.apply(lambda x: x.replace(',',' '))
    return data.head()

def textblob_sentiment_analysis(data, column, score):
    '''
    Function to take in a column name and theshold score that first returns 
    the percentage of data below the threshold and then returns the 
    percentage of data above the threshold.
    
    Input
    -----
    column : str 
        Series object from dataframe
    score : int
        Threshold in question
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Statements about each of the dataframe class subsets indicating the 
    percentage that is above and the percentage that is below a the
    score input
    '''
    # Create dataframe subsets
    anti = data[data.sentiment == -1]
    neutral = data[data.sentiment == 0]
    man = data[data.sentiment == 1]
    news = data[data.sentiment == 2]
    
    # Get percentages for below threshold
    below_perc_anit = round((len(anti[anti[column] < score]) / len(anti)), 3)
    below_perc_neutral = round((len(neutral[neutral[column] < score]) / len(neutral)), 3)
    below_perc_man = round((len(man[man[column] < score]) / len(man)), 3)
    below_perc_news = round((len(news[news[column] < score]) / len(news)), 3)
    
    # Get percentages for above threshold
    above_perc_anit = round((len(anti[anti[column] > score]) / len(anti)), 3)
    above_perc_neutral = round((len(neutral[neutral[column] > score]) / len(neutral)), 3)
    above_perc_man = round((len(man[man[column] > score]) / len(man)), 3)
    above_perc_news = round((len(news[news[column] > score]) / len(news)), 3)
    
    # Printing results
    print('{}% of the anti man-made data is below the {} threshold of {}'.format(below_perc_anit, column, score))
    print('{}% of the neutral data is below the {} threshold of {}'.format(below_perc_neutral, column, score))
    print('{}% of the man-made data is below the {} threshold of {}'.format(below_perc_man, column, score))
    print('{}% of the news data is below the {} threshold of {}'.format(below_perc_news, column, score))
    print('\n')
    print('{}% of the anti man-made data is above the {} threshold of {}'.format(above_perc_anit, column, score))
    print('{}% of the neutral data is above the {} threshold of {}'.format(above_perc_anit, column, score))
    print('{}% of the man-made data is above the {} threshold of {}'.format(above_perc_anit, column, score))
    print('{}% of the news data is above the {} threshold of {}'.format(above_perc_anit, column, score))
    

def element_present_plot(column_name, element, title):
    '''
    Function to checks the number of a specified element in the 
    dataframe subset and returns a plot showing the rate at 
    which the element appears in the subset
    
    Input
    -----
    column_name : str 
        Any name that indicates the element is present
    element : str
        Specific element you want to check for
        Ex: 'http'
    title : str
        Name of element for plot
        Ex: 'Hyperlink'
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Plot showing the rate at which the element appears in each of
    the indvidual subsets
    '''  
    # Creating column column
    data[column_name] = data.message.apply(lambda x: 1 if element in x else 0)

    # Resetting class subsets to include new column
    anti = data[data.sentiment == -1]
    neutral = data[data.sentiment == 0]
    man = data[data.sentiment == 1]
    news = data[data.sentiment == 2]

    # Specifying y-values
    element_frequencies = ((anti[column_name].sum() / len(anti)), 
                           (neutral[column_name].sum() / len(neutral)),
                           (man[column_name].sum() / len(man)),
                           (news[column_name].sum() / len(news)))

    # Building graph
    plt.figure(figsize=(20,10))
    sns.barplot(class_labels, element_frequencies)
    plt.title('{} Frequency by Class'.format(title), fontsize=13, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('{} in Tweet Rate'.format(title))
    
    return plt.show()


def element_count_plot(column_name, element, title):
    '''
    Function to checks the number of a specified element in the 
    dataframe subset and returns a plot showing the average number 
    of times that element appears per tweet for each subset
    
    Input
    -----
    column_name : str 
        Any name that indicates the element is present
    element : str
        Specific element you want to check for
        Ex: 'http'
    title : str
        Name of element for plot
        Ex: 'Hyperlink'
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Plot showing the average number of times that element shows
    up per tweet for each subset
    '''  
    # Creating column column
    data[column_name] = data.message.apply(lambda x: x.count(element))

    # Resetting class subsets to include new column
    anti = data[data.sentiment == -1]
    neutral = data[data.sentiment == 0]
    man = data[data.sentiment == 1]
    news = data[data.sentiment == 2]

    # Specifying y-values
    element_count_means = ((anti[column_name].mean()),
                           (neutral[column_name].mean()),
                           (man[column_name].mean()),
                           (news[column_name].mean()))

    # Building graph
    plt.figure(figsize=(20,10))
    sns.barplot(class_labels, element_count_means)
    plt.title('Average Number of {}s Per Tweet'.format(title), fontsize=13, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Average'.format(title))
    
    return plt.show()

def check_uppercase(data):
    '''
    Function to check if there are any uppercase words in a 
    list of words.
    
    Input
    -----
    data : list (str)
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Binary output indicating presence of uppercase word
    '''
    new_data = 0
    for word in data:
        if word.isupper():
            new_data = 1
        else:
            pass
    
    return new_data


def word_associations_plot(load_association_list,title):
    '''
    Function to plot the rate at which words from one list
    appear in another list
    
    Input
    -----
    load_association_list : list (str)
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Barplot of rate of words in given list
    '''   
    # Create dataframe subsets
    anti = data[data.sentiment == -1]
    neutral = data[data.sentiment == 0]
    man = data[data.sentiment == 1]
    news = data[data.sentiment == 2]
    
    # Set word list equal to loaded list
    word_list = load_association_list
    
    # Tokenize word_list
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    data.message = data.message.apply(lambda x: tokenizer.tokenize(x))
    
    # Lowercase word_list
    data.message = data.message.apply(lambda x: lowercase(x))
    
    # Untokenizing data
    data.message = data.message.apply(lambda x: ','.join(x))
    data.message = data.message.apply(lambda x: x.replace(',',' '))
    
    # Combing all text for each class
    anti_full_text = " ".join(tweet for tweet in anti.message)
    neutral_full_text = " ".join(tweet for tweet in neutral.message)
    man_full_text = " ".join(tweet for tweet in man.message)
    news_full_text = " ".join(tweet for tweet in news.message)
    
    # Anti class counter
    words_anti = 0
    
    # Counting
    for word in word_list:
        if word in anti_full_text:
            words_anti += 1
            
    # Rate for anti class
    anti_word_rate = (words_anti / len(anti))
    
    # Neutral class counter
    words_neutral = 0

    # Counting
    for word in word_list:
        if word in neutral_full_text:
            words_neutral += 1

    # Rate for neutral class
    neutral_word_rate = (words_neutral / len(neutral))

    # Man class counter
    words_man = 0

    # Counting
    for word in word_list:
        if word in man_full_text:
            words_man += 1

    # Rate for man class
    man_word_rate = (words_man / len(man))

    # News class counter
    words_news = 0

    # Counting
    for word in word_list:
        if word in news_full_text:
            words_news += 1

    # Rate for news class
    news_word_rate = (words_news / len(news))

    # Defining y-values
    word_rate = (anti_word_rate,
                 neutral_word_rate,
                 man_word_rate,
                 news_word_rate)
    
    # Plotting bar graph
    plt.figure(figsize=(20,10))
    sns.barplot(class_labels, word_rate)
    plt.title('{} Associated Word Rate'.format(title), fontsize=22, fontweight='bold')
    plt.xlabel('Class', fontsize=15, fontweight='bold')
    plt.ylabel('Rate', fontsize=15, fontweight='bold')
    
    return plt.show()


def lowercase(word_list):
    '''
    Function to lowercase all words in a list.
    
    Input
    -----
    data : list (str)
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Same input list now lowercased
    '''
    lowered = []
    for x in word_list:
        x = x.lower()
        lowered.append(x)
    return lowered

def word_association_features(data, word_association_list):
    '''
    Function to count the number of words in an input list
    that appear in another list.
    
    Input
    -----
    data : list (str)
    wordassociation_list : list (str)
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Count of how many words from data appear in word_association_list
    '''
    count = 0
    for word in data:
        if word in word_association_list:
            count += 1
    
    return count

def simple_custom_features(data):
    
    '''
    Function to create custom features for an existing dataframe.
    
    Input
    -----
    data : Pandas Dataframe
    
    Optional Input
    --------------
    None
        
    Output
    ------
    New DataFrame Columns:
        textblob_polarity - textblob polarity score for message column value
        textblob_subjectivity - textblob subjectivity score for message column value
        tweet_length - length of message column value
        hyperlink_present - binary for presence of hyperlink in message column value
        retweet_present - binary for presence of retweet in message column value
        mention_present - binary for presence of mention in message column value
        mention_count - number of mentions present in message column value
        hashtag_present - binary for presence of hashtag in message column value
        hashtag_count - number of hashtags present in message column value
        exclamation_point - binary for presence of exclamation point in message column value
        question_mark - binary for presence of question mark in message column value
        dollar_sign - binary for presence of dollar sign in message column value
        percent_symbol - binary for presence of percent symbol in message column value
        colon - binary for presence of colon in message column value
        semi_colon - binary for presence of semi-colon in message column value
    '''
    data['textblob_polarity'] = data['message'].apply(lambda x: TextBlob(x).sentiment.polarity)                                                        
    data['textblob_subjectivity'] = data['message'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    data['tweet_length'] = data['message'].apply(lambda x: len(x))
    data['hyperlink_present'] = data['message'].apply(lambda x: 1 if 'http' in x else 0)
    data['retweet_present'] = data['message'].apply(lambda x: 1 if 'RT' in x else 0)
    data['mention_present'] = data['message'].apply(lambda x: 1 if '@' in x else 0)
    data['mention_count'] = data['message'].apply(lambda x: x.count('@'))
    data['hashtag_present'] = data['message'].apply(lambda x: 1 if '#' in x else 0)
    data['hashtag_count'] = data['message'].apply(lambda x: x.count('#'))
    data['exclamation_point'] = data['message'].apply(lambda x: 1 if '!' in x else 0)
    data['question_mark'] = data['message'].apply(lambda x: 1 if '?' in x else 0)
    data['dollar_sign'] = data['message'].apply(lambda x: 1 if '$' in x else 0)
    data['percent_symbol'] = data['message'].apply(lambda x: 1 if '%' in x else 0)
    data['colon'] = data['message'].apply(lambda x: 1 if ':' in x else 0)
    data['semi_colon'] = data['message'].apply(lambda x: 1 if ';' in x else 0)
    
    return data.head()

def load_news_words():
    '''
    Function to load news words.
    
    Input
    -----
    data : list (str)
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Same input list now lowercased
 
    Source
    ------
    https://relatedwords.org/
    '''
    news_words = "fresh,modern,young,newly,late,early,novel,original,inexperienced,recent,current,newborn,recently,other,unused,parvenu,revolutionary,newfangled,linguistics,baby,untried,freshly,raw,unexampled,rising,hot,parvenue,radical,untested,unprecedented,refreshing,virgin,newfound,sunrise,age,unaccustomed,brand-new,bran-new,new-sprung,red-hot,next,similar,will,another,already,newness,newbie,for,on,now,its,as,well,also,which,york,anew,youthful,plans,the,part,new-fangled,inexperient,unweathered,in,addition,to,same,this,making,youngish,with,first,has,it,today,one,and,based,plan,both,business,includes,end,from,move,setting,word,own,including,made,set,unlike,created,key,public,home,example,instead,a,focus,although,an,future,called,included,washington,include,introduced,that,take,america,time,major,opened,came,though,would,instance,while,but,once,way,make,beginning,full,only,working,most,where,post,planned,moving,put,work,american,country,of,week,house,still,immature,newcomer,adolescent,youngness,junior,adolescence,youngly,novelty,infantile,'s,brand new,youth,modernity,reinvigorate,youngth,immaturity,renew,younghood,recency,juvenile,youngster,refurbish,youngling,ignorant,unaccustomed to,puny,novice,neo,teenager,infancy,newmodel,babyishfreshness,premature,naive,vernal,unyoung,infant,newfront,boyish,prematurely,unfamiliar,childish,refresh,former,undeveloped,neonate,babe,babyhood,beaverling,foundling,saxophonist,rejuvenate,newie,nascent,earlyishageless,minikin,puerile,childism,elfin,innocent,bantling,childlike,underage,bantam,previous,turkeyling,preteen,young fogey,young blood,erstwhile,secondhand,hatchling,precocious,eldest,unprocessed,houndling,previously,swanling,minority,rugrat,childhood,little,brat,littleness,babyless,youthless,firstborn,minor,archaic,adolesce,maturity,teenage,novelette,petty,bambino,smallish,dwarfish,prior,small,olden,tidings,intelligence,green,old,revamped,existing,groundbreaking,latest,redesigned,unique,unveil,innovative,expanded,changing,additional,revitalized,additions,streamlined,different,reworked,modernized,introduce,renewed,upcoming,refreshed,expansion,redesign,reconfigured,remodeled,reinvigorated,futuristic,permanent,overhauled,introduction,soontweak,mporary,flagship,refurbished,reshaping,fangled,nursling,bearling,quaint,born,dinky,pre,toddler,mini,lastborn,unked,earliness,unaged,babyship,young animal,younghead,pantywaist,ingenu,pusil,gusu,newname,sproglet,fogey,babysat,farrow,newsworthiness,unworn,spiffy,snazzy,rethinks,obsoleting,supersmall,babygro,ultrasmall,unaging,ageism,newform,age group,young bird,young adult,little old,young mammal,fresh start,big baby,cry loudly,very young child,middle child,wet behind ear,young human,young fish,small person,page boy,baby boy,middle age,crib lizard,come of age,small child,baby food,time of life,old fashion,child carrier,little one,age reversal,feed bottle,old blighty,mother and father,new to,news show,news program,wet behind the ears,newly arisen,cutting edge,clean slate,up to date,old farand,born again"
    news_words = news_words.split(',')
    return news_words

def load_climate_change_words():
    '''
    Function to load climate change words.
    
    Input
    -----
    data : list (str)
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Same input list now lowercased
 
    Source
    ------
    https://relatedwords.org/
    '''
    climate_change_words = "change,earthward,depolarization,sunlight,mutation,nationalize,revolutionize,earthly,transformdestabilize,nationalization,terran,earthy,terrestrial,earthen,scientist,changeful,sublunary,mutate,alteration,alter,unchanging,transmute,borehole,immortalize,terra,changer,immutable,flora,liberalize,gaea,terrene,fauna,transformation,deaden,conversion,secularization,convert,transmutation,transformational,innovate,diversify,dinge,diversification,modifiable,unchanged,demotion,chasten,louden,transformer,barbarize,vesiculate,replace,everchanging,steepen,earthican,denaturalize,earther,planetary,earthbound,dynamize,decimalization,nazify,earthling,professionalize,suburbanize,etherealize,plasticize,stiffen,overchange,deaminate,immutationeroticize,europeanize,unscramble,sentimentalize,classicize,earthian,assibilate,transitivize,downshift,dissimilate,archaize,inactivate,transchange,remew,immaterialize,metamorphosis,coarsen,incandesce,earthman,embrittle,our,modify,keratinize,symmetrize,earthlight,stabilize,decimalize,brutalize,earthless,opacify,conventionalize,normalize,denature,radicalize,flocculate,uniformize,converter,changelog,paganize,tellurian,hydrolyze,modernize,vascularize,peron,orientalize,transfigure,earthscape,metamorphic,democratize,world,transaminate,vulgarise,orogeny,communization,earthlike,changes,earthwoman,caseate,gelatinize,desensitize,islamize,ize,unsanctify,transmogrify,transpeciate,earthboard,conversive,subsoil,intervary,dehydrogenate,geo,entitize,terrestrially,ground,earthshine,reflate,assimilate,glamorize,exoterrene,slenderize,filtration,kaleidoscopic,impact,replacement,transearth,geoengineering,glebe,unchangeable,obsolesce,periglaciation,shortchange,monopole,sorcerize,alternation,changing,makeover,economic,depolarize,creolize,future,masculinize,activate,environment,oxidise,modification,focus,opalize,metamorphose,diatomite,unearth,variation,emulsify,conditions,switch,vitalize,terraform,affect,deepen,policy,global climate change,current,seismic,edit,atmosphere,earthborn,situation,opsonize,exchanger,concerned,earthsman,terraforming,urbanize,this,terraceous,ulcerate,complexify,automatize,progress,regress,effect,convertee,extrasolar,difficult,blur,counterchange,pedosphere,achromatize,possible,transition,recombine,continue,deformation,likely,problem,bestialize,concerns,ways,decarboxylate,approaches,should,suggests,suggest,versicolour,maunder minimum,orbiter,step,measures,critical,alkalinize,view,challenges,rarefy,possibility,volatilize,atterration,necropanspermia,outmode,changeset,need,see,implications,depends,beyond,mean,stability,continues,earthrise,issues,result,agenda,soil,follow,means,concern,particular,issue,cambist,electrically,clear,create,further,alchemize,unlikely,hemisphere,continuing,creating,particularly,better,reflect,globally,especially,noting,crisis,very,indeed,strategy,moreover,decrepitate,overskies,needs,industrialize,trend,reflects,consider,understratum,approach,acetylate,important,consensus,term,context,move,suggested,yet,reason,that,will,idea,measure,arterialize,expect,process,uncertainty,demythologize,due,differences,developing,deodorize,gaia,focused,meant,debate,fact,consequences,would,policies,untunemythologize,allegorize,uglify,extraplanetary,decentralize,loam,magnetosphere,earthworm,isomerize,biodiversity,desertification,emissions,carbon,anthropogenic,climatic,environmentalism,pollution,ecosystems,acidification,ecology,oceans,sustainability,conservation,overpopulation,overfishing,globalization,climat,glaciers,climatology,emitters,biofuel,terrorism,droughts,urbanization,pandemic,protectionism,disasters,wmo,boreal,agriculture,forests,multilateral,obesity,tuvalu,conservationists,malthusian,eco,freshwater,impacts,catastrophe,humankind,habitats,fisheries,eutrophication,epidemics,pastoralism,mitigation,greening,geopolitics,forestry,naturalize,unearthly,demagnetize,depersonalize,clod,earthbag,internationalize,eartheater,vulcanize,reorient,vivify,atmospheric physics,geography,grind,weathering,synoptic scale meteorology,alterable,tectonic,decalcify,topsoil,mesoplanet,earthship,precession,apollo,cryosphere,one moon,7 continent,our planet,blue planet,seven continent,saponify,oblate spheroid,land,extreme weather,tumulus,vitrify,this world,presto change o,earthhole,territorialize,presto chango,overland,libration,worldwide,sol iii,plate tectonics,geosphere,militarize,globe,sahara,volcanic eruptions,dirt,exoatmospheric,dojin,moorland,earthbank,geophagic,equatorial,ipcc,trioctile,geoheliocentrism,interplanetary,co2,change order,landward,mother earth,sun,live on,geospace,asthenosphere,energy,decarbonise,decarbonization,decarbonizing,sex change,catastrophism,polluter,undernutrition,environmentalists,biopiracy,gigaton,intermittency,paleoclimatology,icecaps,glaciology,alarmism,enviro,overexploitation,overexploit,geodynamics,peatland,deglaciation,anthropocentrism,scarcities,ecocide,salination,diatomaceous earth,geothermal,change intensity,translunar,evapotranspiration,archean,general circulation model,temperature change,outline of physical science,middle earth,sea change,central tendency,earth stop,statistical variability,muckland,super earth,landly,el niño,soillesshell on earth,upmass,planetfall,environmental policy,podzol,lot of water,heliocentrism,planetscape,nammu,human impact on the environment,modulation,paleoproterozoic,hydrospace,skywave,climate forcing,cyclostratigraphy,unsoiling,grindingly,preground,seism,astrometeorology,geosynclinal,cosmozoa,change up,epeirogenesis,come round,change of direction,duneland,sublunar,cern,bogland,proxigean spring tide,regosol,groundable,mohole,merland,many animal,nature,life zone,climate feedback,volcano,ecological threshold,ton,thermohaline circulation,volcanic ash,get change,stratosphere,thermal expansion,russell's teapot,terrestrial planet,many country,earth's atmosphere,change one's mind,sublunary sphere,de emphasize,change taste,earth science,loss of consciousness,break into,lunar phase,atmosphere of earth,change by reversal,our world,volumetric heat capacity,little ice age,milky way galaxy,earth fast,goldilocks planet,space communication,space debris,pacific decadal oscillation,inner solar system,habitable zonegoldilocks zone,our solar system,north atlantic oscillation,inner planet,big place,old earth creationism,arctic oscillation,inferior planet,lunar distance,lunar eclipse,water cycle,greenhouse emission,acid rain,carbon dioxide,air pollution,hydrologic cycle,soil erosion,ozone layer,bark beetle,gaia hypothesis,urban sprawl,ursus maritimus,quaternary period,russian federation,malthusian theory,atlantic,space satellite,pacific,place name,carboniferous,orbital eccentricity,supercontinent,terra firma,axial tilt,blue green alga,pangaea,solar system,seven sea,thermal lithosphere,island,hubble law,glacial period,chemical lithosphere,interglacial period,solar eclipse,solar year,major planet,particulate,lunar year,cement,carbon planetdesert soil,geologic record,pole star,nine planet,off world,glebe land,silicate planet,microclimate,grind tissue,trans neptunian,thermal inertia,force land,supervolcano,dry land,vegetation,very large,faint young sun paradoxpliocene,great oxygenation event,red giant,white dwarf,archaeological,solar variation,solar cycle,glacier,spörer minimum,ice age,anno domini,radiative forcing,interglacial,holocene,dendrochronology,sulfur dioxide,interpolation,beetle,limestone,tephra,arctic,satellite,altimeter,evaporation,pollen,fish,sulfuric acid,radiosonde,mount pinatubo,mount tambora,year without a summer,large igneous province,flood basalt,mass extinction,prediction of volcanic activity,climate model,carbon dioxide sink,us geological survey,toba catastrophe theory,isthmus of panama,western boundary current,dendroclimatology,palynomorph,palynology,tephrochronology,ooids,autotrophs,gulf stream,scientific opinion on climate change,ozone depletion,ice cap,sea level change,glacial geology,instrumental temperature record,satellite temperature measurements,oxygen isotope ratio cycle,oral history,historical documents,mass balance,retreat of glaciers since 1850,glacier mass balance,north pole,continental climate,heinrich event,dansgaard–oeschger event,younger dryas,southern ocean,quaternary glaciation,carboniferous rainforest collapse,coral reef,marine terrace,ice sheet,orbital forcing,uranium-thorium dating,radiocarbon dating,cosmogenic radionuclide dating,antarctic ice sheet,atlantic period,primary productivity,polar desert,tide gauge,last glacial maximum,climate,global warming,earth,biosphere,global,weather,greenhouse gas,albedo,lithosphere,shift,warming,environmental,hydrosphere,continental drift,planet,solar radiation,proxy,hadean,moraine,ice core,deforestation,cloud,fossil fuel,el niño-southern oscillation,carbon cyclemilankovitch cycles"
    climate_change_words = climate_change_words.split(',')
    return climate_change_words

def load_democratic_party_words():
    '''
    Function to load democratic party words.
    
    Input
    -----
    data : list (str)
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Same input list now lowercased
 
    Source
    ------
    https://relatedwords.org/
    '''
    democratic_party_words = "party,political party,democrat,democracy,sdp,multiparty,fiesta,partyism,partyness,jamboree,nonparty,shindig,partyer,partymeister,intraparty,preparty,afterparty,housewarming,kuomintang,counterparty,major party,cross party,festivity,partygoer,bunfight,slumber party,soiree,birthday celebration,festive,fete,hen party,after party,surprise party,stag party,celebration,party hat,war party,bachelorette party,party colour,party dress,search party,cocktail party,bachelor party,person dance,sociable,social gather,celebratory,reunion,third party,political,tailgate party,social,ceilidh,partisan,keg party,house party,eat cake,pool party,democrats,revelry,conservative party,merrymaking,birthday,noncelebration,gala,birthdayless,social butterfly,crossbencher,unbirthday,celebrate birthday,partyware,sweet seventeen,beano,sociality,celebrate,politics,jubilate,bridal shower,social occasion,tammany,pinata,tea party,party puffer,eighteenth,privacy policy,partygoing,raver,bash,anniversary,political system,open house,government,drunkeness,uncelebrated,celebrator,celebratedly,festival,twenty first,leave do,celebrant,assignor,republican,blow candle,annual event,annual celebration,half birthday,special day,quadripartite,demoparty,tammany hall,invite guest,tammany society,dpp,special occasion,costume party,get gift,invite friend,election,get together,liberal,birthday party,elections,have friend over,send out invitation,socially,gop,vote,person gather,party line,invite person,public holiday,political opposition,party whip,barmy army,dorothy dixer,go off reservation,wed party,work majority,governance,opposition,candidate,coalition,conservative,revel,parties,famously,presidential,leader,leadership,conservatives,candidates,majority,ruling,parliamentary,incumbent,socialist,centrist,elected,legislative,caucus,republicans,polls,leaders,alliance,legislators,presidency,lawmakers,support,congress,re,campaign,nationalist,voters,campaigning,supported,senate,opposed,minority,reform,parliament,progressive,liberals,faction,supporters,politicians,senator,labour,votes,gore,representatives,voting,candidacy,independence,socialists,president,member,governing,likud,backed,pro,dole,legislature,campaigned,endorsed,kmt,communist,leftist,outgoing,cabinet,communists,bush,ndp,favor,allies,elect,opponents,clinton,declared,vowed,electoral,agenda,supporter,obama,labor,assembly,independent,mccain,supporting,solemnize,organization,merrymake,governmental,interpol,organisation,politically,jolly,bureaucracy,rave,bilateral,doof,federation,organizational,polity,fiefdom,socialism,privatization,organise,socialization,jubilee,wrecker,quorum,socialize,birthdate,commemoration,dos,dp,luminary,organize,disorganization,uango,gymnopaedia,event,cake,wedding,fizzer,anarchy,democratic,organigram,territorialization,preorganization,subsidiary,politic,corporate,freemasonry,conservancy,ombudsman,partywear,hierarchy,systematization,manifesto,corporatism,establishment,impleader,corporation,treasurer,reorganize,anarchism,nasa,contractee,rebirthday,vouchee,bridecake,have party,pick up piece,unionisation,collectivize,gateau,tharcake,gâteau,caky,throw party,name day,person cheer,menshevik,donkey,watergate,adhocracy,bolshevik,whig,lassalle,paint town red,spd,sdlp,liebknecht,bebel,date of birth,independence day,special event,case of beer,for birthday,christmas card,feast day,get present,sausage party,happy person,self organization,birthday cake,unite nation,life of party,party pooper,status conference,secret society,dixiecrats,international olympic committee,locofoco,hunker,barnburner,you get drunk,meet new person,change management,private sector,happy birthday,check and balance,nation of islam,orange order,fannie mae,bake cake,drink beer,nast,populist,rep,drink alcohol,populism,jane doe,goody bag,celebrate holiday,zaire,yemen,jeffersonian,rota,republic,mink,idea,representative,rec,public,free,drc,ron,congo,rcd,participation,congolese,madagascar,chile,collective,japan,blow off steam,silver,carter,cox,ballot,wilson,dean,bolt,bolter,boss,bandwagon,adherence,adhesion,administration,apostasy,apostate,attachment,bachelor,advantage,belle,aides,bloc,give gift,win baseball game,drink too much,world bank,hold rein,baby shower,escape clause,go to party,john doe,data warehouse,terrorist organization,hen night,become intoxicate,world organization,wed cake,dinner party,group of person,social group,hold company,hostile witness,feel good factor,sheet cake,meet person,thirsty thursday,girl scout,commonwealth of nation,adverse party,political orientation,party state,labor union,off message,system engineer,net raise,board of director,trade union,social control,administrative unit,legal entity,enterprise architecture,self-government,democratize,timor-leste,alist,republican party,whig party,democratic-republican party,american labor party,social democratic party,social democrat,social democracy,new democratic party,solid south,socialist party,east germany,north korea,free world,sri lanka,southern yemen,anti-masonic party,american party,american federalist partyblack panther"
    democratic_party_words = democratic_party_words.split(',')
    return democratic_party_words

def load_republican_party_words():
    '''
    Function to load republican party words.
    
    Input
    -----
    data : list (str)
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Same input list now lowercased
 
    Source
    ------
    https://relatedwords.org/
    '''
    republican_party_words = "gop,party,senate,political party,majority,saudis,pataki,harrelson,newsom,kinnear,polanski,armenians,fiesta,multiparty,jamboree,baros,colly,celebration,fete,festive,democratic,festivity,republican,kuomintang,celebratory,soiree,revelry,shindig,nonparty,partyer,preparty,intraparty,partyism,afterparty,partymeister,democrats,partyness,housewarming,merrymaking,candidate,celebrate,birthday,conservative,jubilate,democrat,gala,republicans,partygoer,revel,election,counterparty,conservatives,opposition,liberal,candidates,presidential,incumbent,beano,coalition,famously,sociable,senator,vote,lawmakers,anniversary,noncelebration,dole,leadership,campaign,elections,eighteenth,gore,reunion,political,leader,legislators,voters,bunfight,elected,mccain,ceilidh,caucus,senior,unbirthday,festival,partisan,birthdayless,centrist,polls,parties,congress,campaigning,legislative,celebrant,ruling,bush,social,clinton,parliamentary,obama,liberals,leaders,kerry,nominee,congressional,politicians,votes,supported,opposed,supporters,county commissioner,advisory board,charter school,look forward to,primary election,get through,state senator,come into,house of representatives,white people,bush administration,democratic party,common pleas,school system,city manager,school superintendent,send in,but then,north korean,electoral college,roll back,come out,turn back,figure out,young lady,murder suspect,settle for,find out,judicial system,allow for,with that,air mass,real property,equal opportunity,home rule,stand for,stand on,build on,sum up,walk through,go down,come to,set out,turned out,run off,bring on,cut down,election day,special session,small town,debt ceiling,cut back,put off,factor in,measure up,call off,too little,push for,public opinion,lie about,pretty much,district court,catch up with,stick to,depend on,one another,budget for,personal information,hockey player,district attorney,circuit court,long island sound,take on,school board,national assembly,gay man,property tax,of their own,rein in,stamp out,general assembly,wind up,play on,water conservation,crew member,senate race,housing project,candidacy,bring down,put together,support,bash,presidency,nomination,endorsed,campaigned,supporter,gingrich,nationalist,legislature,opponent,minority,seniority,opponents,independents,parliament,socialist,backed,congressman,reform,gephardt,celebratedly,likud,representatives,celebrator,labour,uncelebrated,favor,tory,uchanan,alliance,voting,senators,president,politics,lieberman,member,voted,barack,pro,dpp,faction,gubernatorial,progressive,gramm,romney,outgoing,lawmaker,allies,partyware,sociality,pinata,drunkeness,solemnize,elder,partygoing,raver,merrymake,slumber party,birthday celebration,cross party,after party,surprise party,old,party hat,sdp,oldish,hen party,eldest,assignor,stag party,ancient,tailgate party,jolly,cocktail party,war party,elderly,party dress,antiquary,person dance,centenarian,antediluvian,antiquarian,rave,immemorial,eat cake,paleo,octogenarian,bachelorette party,anciently,bilateral,sen.,eld,olden,keg party,doof,age,bachelor party,venerable,ould,geriatric,emeritus,party colour,antiquity,demoparty,antique,pool party,major party,senile,organization,teenage,musty,celebrate birthday,search party,house party,organisation,ex,elden,ageless,ery,obsolete,third party,jubilee,hoary,sweet seventeen,interpol,archaic,socialization,wrecker,outdated,social gather,commemoration,quadripartite,socialize,institution,luminary,crossbencher,underage,organizational,curmudgeon,mesolithic,paleolithic,socially,bureaucracy,gymnopaedia,cake,erstwhile,dotage,former,event,secondhand,wedding,senescent,institutional,party puffer,aad,social butterfly,elderish,oldling,fizzer,eldern,ancientry,longtime,overage,organise,social occasion,twenty first,vetust,archæ,methuselah,oldly,brokerage,hierarchy,oldness,anile,cthulhu,cohort,ancientness,ancienty,oldster,antic,inductee,nonagenarian,conservative party,ageful,inveterate,federation,organize,fogey,agedness,antiaging,blow candle,nonaged,ageist,nonaging,grandevous,partywear,bridal shower,unaging,oldbie,annual event,ripe,youthless,annual celebration,half birthday,leave do,centurion,tea party,ageism,disorganization,quadragenarian,establishment,institute,open house,antiquate,impleader,special day,invite guest,special occasion,get gift,sexagenarian,contractee,preorganization,unaged,rebirthday,unyoung,catabiosis,bridecake,vouchee,quango,evenold,invite friend,territorialization,senesce,have party,costume party,pick up piece,yeared,gateau,tharcake,gâteau,teacherage,caky,wrinkly,dought,birthday party,send out invitation,throw party,organigram,person cheer,name day,systematization,vicenarian,paleography,madrina,independence day,invite person,staleness,coetaneous,have friend over,paint town red,elderdom,unked,get together,unionisation,person gather,date of birth,special event,case of beer,for birthday,public holiday,christmas card,get present,feast day,sausage party,happy person,birthday cake,party line,over old,life of party,old guard,good old,age old,rotary telephone,old norwegian,old icelandic,party pooper,status conference,privacy policy,old timer,senior citizen,long in tooth,some old,how old,any old,you get drunk,old fogey,old fashion,mental age,old fart,age reversal,meet new person,historic period,happy birthday,old church slavonic,atomic age,wed party,party whip,bake cake,old lace,drink beer,drink alcohol,old flame,old latin,go off reservation,drink age,jane doe,aad wife,celebrate holiday,goody bag,old dutch,old saxon,big old,dorothy dixer,old italian,barmy army,blow off steam,year old,fannie mae,win baseball game,middle age,political opposition,old age,escape clause,drink too much,give gift,baby shower,old folk,political system,international olympic committee,go to party,wed cake,become intoxicate,hen night,gray haired,over hill,john doe,age group,sheet cake,off message,dinner party,hostile witness,self organization,thirsty thursday,group of person,old norse,out of date,old high german,old prussian,of age,meet person,ivy league,may december,feel good factor,adverse party,little old,old school,orange order,old thing,age of consent,unite nation,old time,world bank,old blighty,old timey,school age,age of reason,labor union,come of age,old see,secret society,old frisian,arcus senilis,old hat,business intelligence"
    republican_party_words = republican_party_words.split(',')
    return republican_party_words

def load_tier1_words():
    '''
    Function to load tier 1 words.
    
    Input
    -----
    data : list (str)
    
    Optional Input
    --------------
    None
        
    Output
    ------
    List containing all element in tier one words
 
    Source
    ------
    http://soltreemrls3.s3-website-us-west-2.amazonaws.com/marzanoresearch.com/media/documents/List-of-Tier-1-Basic-Terms.pdf
    '''
    tier_one_words = "can, cannot, could, may, might, must, shall, should, will, would, as, at, during, now, of, on, together, when, while, did, do, does, doing, done, had, has, have, am, are, be, been, is, was, were, being, and, of, too, with, [he, him, I, it, me, myself, she, them, they, us, we, you, her, hers, its, mine, my, our, their, your, yours, his, ours, theirs, [what, when, where, which, at, from, to, [because, by, for, from, if, since, so, then, to, because of, that, which, who, how, why, a, an, each, every, no, that, the, these, this, those, either, ah, aha, bye, gee, good-bye, ha, hello, hey, hi, ho, maybe,no, oh, ok, okay, ooh, wow, yes, goodnight, wow, more, most, much, so, such, sure, too, very, well, badly, [already, early, fresh, new, ready, since, young, ago, lately, left, right, east, north, south, west, almost, enough, just, only, hardly, alone, mostly, nearly, simply, all, another, both, few, half, less, little, lot, many, more, most, none, only, other, pair, two, whole, amount, couple, extra, several, single, twice, along away, beside, between, by, close, far, near, past, toward, apart, aside, beyond, nearby, opposite, outer, ahead, back, behind, end, forward, front, middle, center, last, ahead of, among, backward, backwards, rear, across, in, inside, into, out, outside, through, enter, outdoors, indoor, indoors, throughout, within, [below, bottom, down, low, under, beneath, underneath, downhill, downstairs, downward, before, late, next, soon, then, until, afterward, afterwards, later, latter, here, there, where, nowhere, somewhere, anywhere, someplace, above, high, off, on, over, tip, top, up, onto, upon, aboard, overheard, upright, upside-down, upstairs, upward, but, else, not, or, still, than, without, yet, against, compare, either, except, instead, neither, unless, whether, [eight, five, four, nine, one, seven, six, ten, three, two, zero eighteen, eighty, eleven, fifteen, fifty, first, forty, fourteen, hundred, nineteen, ninety, number, numeral, second, seventeen, seventy, sixteen, sixty, thirteen, thousand, twelve, twenty, billion, decimal, dozen, million, ninth, seventh, sixth, tenth, third, April, August, December, February, Friday, January, July, June, March, May, Monday, November, October, Saturday, September, Sunday, Thursday, Tuesday, Wednesday, maybe, possibly, hopefully, please, bird, chicken, crow, duck, eagle, fowl, goose, hen, jay, owl, parrot, robin, rooster, turkey, big, giant, great, huge, large, little, small, tiny, enormous, gigantic, jumbo, any, each, enough, nothing, some, nobody, anybody, anyone, anything, no one, somebody, someone, something, bunny, calf, cub, kitten, pup, puppy, tadpole, bush, flower, plant, tree, vegetation, weed, corner, edge, limit, margin, side, catch, pass, throw, toss, climb, lift, raise, order, rank, rise, do, use, happen, occur, have, belong, own, possess, they're, we're, you're, sad, sorry, unhappy, bring, carry, deliver, get, give, mail, move, place, present, put, return, send, set, take, bear, remove, [fun, glad, happy, joke, jolly, joy, merry, play, please, silly, celebrate, happiness, humor, joyful, choice, choose, decide, judge, pick, select, appoint, sort, cap, glasses, hat, helmet, hood, mask, sunglasses, crown, breakfast, dinner, lunch, meal, picnic, supper, treat, dessert, address, direction, place, point, position, spot, location, moon, sky, star, sun, universe, world, meteor, planet, space, bite, drink, eat, feed, sip, swallow, chew, age, fall, month, season, summer, week, weekend, winter, year, century, decade, generation, spring, weekday, lullaby, music, poem, rhyme, song, hymn, dance, music, ballet, melody, orchestra, solo, believe, care, enjoy, like, love, forgive, want, [human, individual, people, person, hero, self, black, blue, brown, color, gold, gray, green, orange, pink, purple, red, white, yellow, blonde, colorful, silver, best, better, dear, fine, good, important, perfect, outstanding, super, useful, fast, hurry, quick, race, rush, slow, speed, sudden, dash, slowdown, kindergarten, library, museum, school, classroom, schoolroom, describe, explain, present, say, state, tell, brag, inform, mention, recite, boot, glove, mittens, shoe, skate, sock, stocking, sandal, slipper, [dance, march, run, skip, step, trip, walk, hike, limp, stumble, tiptoe, trot, cat, dog, doggie, fox, lion, tiger, wolf, bulldog, collie, bear, cow, deer, donkey, elephant, giraffe, horse, lamb, pig, pony, rabbit, sheep, bat, bull, kangaroo, moose, raccoon, reindeer, skunk, zebra, go, come, leave, travel, visit, wander, appear, approach, arrive, depart, disappear, exit, journey, proceed, forget, idea, remember, think, thought, wonder, imagine, memory, principal, student, teacher, graduate, pupil, schoolteacher, empty, fill, full, hollow, fish, seal, whale, salmon, shark, tuna, color, copy, draw, paint, print, publish, scribble, sign, spell, write, handwriting, misspell, publish, skim, trace, underline, correct, just, real, right, true, truth, wrong, error, fair, false, fault, honest, mistake, foot, gallon, grade, inch, mile, pound, quart, yard, mouthful, spoonful, tablespoon, dough, flour, gravy, mix, pepper, salt, sauce, sugar, ketchup, mayonnaise, mustard, arm, elbow, finger, hand, thumb, shoulders, wrist, feet, foot, knee, leg, toe, ankle, heel, act, cartoon, film, movie, show, stage, comedy, play, cold, heat, hot, temperature, warm, chill, cool, day, evening, hour, minute, morning, night, noon, second, tonight, afternoon, midnight, overnight, sundown, sunrise, sunset, mouth, teeth, throat, tooth, voice, gum, jaw, lip, tongue, he's, I'm, it's, she's, that's, there's, here’s, what's, where's, alligator, dragon, frog, snake, toad, turtle, dinosaur, mermaid, monster, old, past, present, today, tomorrow, yesterday, ancient, future, history, someday, alarm, bell, horn, phone, doorbell, siren, telephone, he'll, I'll, she'll, they'll, we'll, you'll, butter, cheese, egg, yolk, cream, margarine, beach, island, coast, shore, dentist, nurse, doctor, loss, winner, champion, defeat, win, air, weather, nature, basement, bathroom, cellar, closet, garage, hall, kitchen, nursery, room, bedroom, doorway, hallway, playroom, porch, chain, glue, key, lock, nail, needle, pin, rope, string, cable, knot, screw, shoelace, strap, alley, bridge, driveway, highway, path, railroad, road, sidewalk, street, track, trail, avenue, freeway, mall, racetrack, ramp, route, tunnel, aunt, brother, dad, family, father, granny, ma, mama, mom, mother, papa, parent, sister, son, uncle, cousin, daughter, grandparent, husband, mammy, nephew, niece, sibling, wife, ant, bee, bug, butterfly, caterpillar, fly, insect, ladybug, spider, worm, bumblebee, cockroach, flea, grasshopper, mosquito, moth, slug, wasp, [bowl, cup, dish, fork, glass, knife, pan, plate, pot, spoon, chopsticks, mug, opener, tablespoon, teaspoon, tray, cruise, drive, passenger, ride, row, sail, cruise, glide, gather, group, pile, sequence, bunch, classify, list, organize, stack, deep, height, high, length, long, short, size, tall, thin, wide, depth, narrow, shallow, thick, width, agree, bless, greet, pray, thank, welcome, compliment, cooperate, encourage, praise, ice, rain, snow, water, hail, icicle, liquid, rainbow, raindrop, rainfall, snowball, snowman, steam, [lake, ocean, puddle, river, sea, stream, bay, creek, pond, [hear, listen, loud, noise, quiet, sound, aloud, calm, echo, silence, silent, [cent, coin, dollar, money, penny, quarter, cash, check, dime, nickel, pound, ticket, speak, speech, talk, chat, discuss, statement, cage, cave, shelter, fort, jail, [find, fix, make, build, develop, prepare, produce, repair, shape, branch, leaf, twig, bark, limb, stump, bank, safe, purse, wallet, behave, help, save, heal, improve, protect, girl, lady, woman, female, housewife, schoolgirl, brush, card, crayon, ink, page, paper, pen, pencil, blackboard, chalk, chalkboard, loose-leaf, notebook, paintbrush, bed, bench, chair, crib, desk, drawer, seat, table, bookcase, couch, counter, cradle, cupboard, playpen, sofa, stool, land, lot, place, region, area, location, territory, zone, cheek, chin, face, head, brain, forehead, mind, free, poor, poverty, rich, broke, cheap, expensive, fish, fly, hunt, trap, buck, gallop, soar, sting, oven, radio, stove, television, furnace, heater, fridge, hammer, saw, shovel, tool, drill, rake, screwdrivers, tweezers, balloon, helicopter, kite, plane, rocket, aircraft, airline, airplane, spacecraft, castle, home, hotel, house, hut, apartment, motel, palace, tent, buy, pay, sale, sell, spend, bet, earn, owe, purchase, door, floor, roof, stairs, wall, window, ceiling, doorstep, stair, staircase, stairway, bread, bun, cereal, chips, cracker, crust, hamburger, hotdog, jelly, pancake, pizza, salad, sandwich, snack, toast, biscuit, coleslaw, loaf, macaroni, muffin, noodle, oatmeal, omelet, pretzel, spaghetti, taco, tortilla, waffle, belt, diaper, dress, jeans, pajamas, pants, pocket, shirt, skirt, apron, bathrobe, nightgown, robe, shorts, sweater, tights, long, never, often, once, sometimes, always, anymore, awhile, daily, ever, forever, frequent, hourly, rare, regular, repeat, seldom, twice, usual, weekly, boil, dive, drain, drip, float, melt, pour, sink, spill, splash, stir, swim, wet, bubble, dribble, flush, freeze, leak, slick, slippery, soak, spray, sprinkle, squirt, trickle, bicycle, bike, bus, car, train, tricycle, truck, van, wagon, ambulance, automobile, cab, locomotive, motorcycle, scooter, stagecoach, subway, taxi, taxicab, trailer, fit, fold, sew, tear, wear, braid, patch, rip, wrinkle, zip, bit, dot, flake, part, piece, crumb, member, portion, section, slice, sliver, splinter, typecatch, hold, hug, pick, clasp, cuddle, grab, pinch, snuggle, squeeze, catch, hold, hug, pick, clasp, cuddle, grab, pinch, snuggle, squeeze, asleep, awake, nap, sleep, daydream, dream, pretend, wake, ground, land, mud, soil, clay, dirt, dust, earth, blanket, cover, pillow, towel, bedspread, cushion, napkin, pillowcase, sheet, tablecloth, look, see, stare, watch, blink, peek, spy, wink, bacon, beef, ham, hotdog, sausage, bologna, pork, steak, able, smart, stupid, alert, brilliant, wise, myth, story, fiction, legend, literature, mystery, poetry, riddle, tale, writing, garden, park, yard, patio, playground, schoolyard, ear, eye, nose, eyebrow, eyelash, nostril, drop, fall, lay, dump, slump, tumble, block, rectangle, square, triangle, cube, pyramid, triangular, doll, toy, toys, puppet, puzzle, calendar, clock, watch, date, o’clock, coat, jacket, cape, raincoat, quit, work, hire, labor, begin, start, try, beginning, origin, get, steal, accept, attract, capture, point, wave, clap, handshake, salute, I've, they've, we've, you've, grin, smile, frown, nod, kiss, suck, lick, spit, cake, candy, cookie, cupcake, doughnut, gum, honey, jam, pie, pudding, syrup, brownie, butterscotch, caramel, chocolate, cocoa, fudge, licorice, lollipop, marshmallows, sherbet, sundae, vanilla, coach, direction, know, learn, teach, understand, advice, comprehend, confuse, discover, information, instruct, outsmart, study, suggest, trick, feather, fur, hide, paw, tail, whisker, beak, bill, claw, fin, flipper, hoof, snout, cheer, cry, laugh, roar, shout, sing, whisper, yell, applause, chuckle, cough, giggle, holler, laughter, scream, snore, whistle, yawn, bump, hair, rash, skin, bald, beard, bruise, freckle, pigtail, scar, ball, bat, glove, swing, base, goal, net, softball, touchdown, boat, canoe, ship, raft, submarine, tugboat, yacht, body, lap, neck, belly, chest, hip, waist, hurt, kill, punish, harm, injure, murder, shoot, bake, boil, cook, barbeque, broil, fry, grill, roast, serve, ax, axe, knife, scissors, blade, lawnmower, pocketknife, bag, basket, bath, bathtub, bottle, box, bucket, jar, barrel, coffeepot, container, crate, folder, hamper jug, package, pail, pitcher, sack, suitcase, tub, bang, beep, boom, ring, click, creak, plop, rattle, slam, squeak, toot, zoom, add, count, minus, plus, subtract, addition, cube, divide, division, multiplication, multiply, subtraction, clown, dancer, actor, actress, magician, model, hill, mountain, cliff, hillside, mound, rest, stay, delay, pause, relax, remain, wait, lie, sit, crouch, kneel, squat, find, keep, bury, hide, spot, city, neighborhood, state, town, village, camp, county, downtown, ghetto, heaven, slum, suburb, king, mayor, president, candidate, knight, official, prince, princess, queen, apple, banana, cherry, grape, orange, peach, pear, strawberry, avocado, berry, blueberry, coconut, cranberry, grapefruit, lemon, melon, pineapple, plum, prune, raisin, raspberry, bark, buzz, meow, moo, baa, cluck, gobble, growl, peep, purr, quack, [juice, milk, pop, soup, beer, chili, coffee, soda, stew, tea, wine, answer, ask, call, offer, question, reply, request, respond, test, cloth, rag, thread, cotton, lace, leather, nylon, silk, wool, birthday, party, recess, circus, date, fair, holiday, parade, vacation, country, nation, continent, equator, hemisphere, stick, wood, board, log, post, timber, pull, push, drag, haul, shove, yank, game, recess, contest, race, recreation, sport, show, trade, borrow, lose, loser, share, clean, wipe, rinse, scrub, sweep, wash, pretty, ugly, beautiful, cute, handsome, lovely, fat, heavy, chubby, lean, skinny, slim, mouse, squirrel, beaver, groundhog, hamster, rat, nest, zoo, aquarium, beehive, birdhouse, cocoon, hive, theater, court, gym, stadium, blood, bleed, sweat, grass, lawn, root, vine, flat, even, lean, level, steep, animal, pet, wildlife, appearance, badge, flag, image, scene, sight, view, blow, breath, choke, exhale, hit, slap, spank, touch, beat, feel, knock, pat, pound, smash, tap, tickle, blame, cheat, lie, accuse, argue, complain, dare, disagree, disobey, quarrel, scold, tease, warn, around, roll, turn, clockwise, rotate, spin, surround, swing, twirl, twist, country, family, community, democracy, nation, race, society, tribe, gift, prize, award, medal, reward, savings, treasure, hard, soft, bumpy, firm, rough, smooth, tight, boy, man, guy, hero, male, schoolboy, sir, baby, child, adult, grown-up, kid, teenager, toddler, friend, neighbor, boyfriend, classmate, pal, partner, playmate, bandit, villain, bully, criminal, enemy, killer, liar, pirate, thief, correct, let, obey, advice, allow, command, control, demand, direct, excuse, forbid, force, permit, refuse, remind, require, carrot, corn, nut, peanut, popcorn, seed, almond, bean, cashew, celery, cucumber, lettuce, olive, onion, peas, pickle, potato, pumpkin, rice, spinach, squash, tomato, walnut, wheat, baseball, soccer, softball, swim, swimming, basketball, bicycle, bowling, boxing, football, golf, hockey, racing, skate, skating, ski, skiing, tennis, volleyball, wrestling, grocery, store, bakery, bookstore, cafeteria, drugstore, lunchroom, restaurant, brave, courage, heroic, honest, loyal, button, collar, sleeve, zipper, bone, joint, muscle, skeleton, price, cost, payment, rent, end, complete, finish, last, slip, rock, skid, slide, gate, fence, mailbox, shelf, line, bent, crooked, cross, straight, stripe, alphabet, consonant, letter, symbol, vowel, fire, burn, campfire, flame, spark, easy, difficult, impossible, problem, taste, flavor, juicy, ripe, sour, sweet, tasty, brush, soap, broomstick, floss, mop, shampoo, sponge, suds, toothbrush, toothpaste, brush, comb, handkerchief, buckle, fan, jewelry, kerchief, necklace, perfume, pin, ribbon, ring, scarf, tie, umbrella, news, search, analyze, examine, experiment, explore, homework, investigate, lesson, schoolwork, storm, thunder, blizzard, downpour, draft, hurricane, lightning, thunderstorm, tornado, wind, angel, god, cupid, devil, elf, fairy, ghost, monster, witch, wizard, thankful, considerate, courteous, gentle, grateful, kind, nice, polite, respectful, athlete, batter, boxer, catcher, coach, loser, runner, winner, sick, disease, health, ill, injury, well, pill, aspirin, bandage, medicine, vitamin, hungry, hunger, starve, thirst, thirsty, time, bedtime, daytime, dinnertime, lunchtime, paddle, wheel, anchor, fender, mirror, oar, parachute, seatbelt, tail, tire, trunk, wing, don’t, isn’t, ain't, aren't, can't, couldn't, doesn't, hasn't, haven't, shouldn't, weren't, won’t, wouldn't, job, career, chore, housework, profession, task, worker, rock, boulder, diamond, jewel, marble, stone, word, adjective, adverb, noun, sentence, verb, art, painting, photo, photograph, picture, statue, safe, danger, dangerous, risk, trouble, unsafe, smell, sneeze, sniff, snore, snort, stink, cut, rub, carve, chop, clip, dig, mow, peel, scoop, scratch, shave, slice, snip, stab, bad, awful, evil, terrible, wicked, worse, worst, instrument, banjo, drum, guitar, piano, triangle, violin, dead, alive, born, die, egg, hatch, life, live, wake, food, crop, fruit, meat, seafood, sweets, vegetables, meet, attach, combine, connect, fasten, include, join, marriage, marry,stick, wedding, book, bible, booklet, chapter, cookbook, diary, dictionary, essay, journal, magazine, newspaper, novel, outline, storybook, summary, text, textbook, guess, calculate, clue, compose, conclude, create, design, estimate, fact, information, invent, invention, mystery, prediction, prove, solve, suppose, accident, break, crash, crush, damage, dent, destroy, mark, ruin, scratch, waste, wreck, bar, brick, cardboard, paste, pipe, plastic, sewer, tube, wire, alike, copy, equal, even, example, like, same, similar, twin, athletic, beauty, clumsy, health, might, power, strength, strong, weak, weakness, arrow, bomb, bullet, firecracker, fireworks, gun, sword, advise, appeal, beg, convince, cue, persuade, recommend, suggest, letter, message, note, postcard, poster, signal, valentine, business, law, medicine, military, religion, science, technology, band, class, club, crowd, herd, team, gold, iron, magnet, metal, silver, steel, battle, fight, peace, revolution, war, wrestle, bet, certain, chance, likely, luck, miracle, possible, balance, blank, fancy, order, plain, simple, clothes, clothing, costume, suit, uniform, artist, choir, drummer, painter, singer, firefighter, officer, policeman, sheriff, soldier, minister, nun, pastor, pope, priest, hole, canyon, ditch, manhole, pit, valley, cork, cover, flap, lid, mask, berry, blossom, dandelion, rose, seed, circle, bend, curl, curve, loop, oval, round, twist, bright, clear, light, shiny, sunshine, candle, candlestick, lamp, light, lightbulb, cause, change, effect, outcome, purpose, reason, result, he'd, I'd, she'd, they'd, you'd, battery, brake, engine, jet, motor, computer, keyboard, monitor, mouse, robot, goal, plan, subject, topic, certain, confident, hopeful, proud, sure, diagram, drawing, graph, map, action, activity, motion, play, juggle, shake, shiver, vibrate, wiggle, bounce, fidget, snap, wag, blast, expand, explode, magnify, spread, banner, carpet, curtain, rug, vase, certainly, honestly, really, seriously, simply, truly, comma, language, period, vocabulary, dizzy, fever, itch, pain, garbage, junk, litter, trash, common, familiar, normal, ordinary, popular, regular, usual, odd, rare, special, strange, weird, afraid, alarm, fear, nervous, anger, angry, dislike, hate, mad, expect, miss, need, selfish, want, wish, active, busy, eager, responsible, crazy, mad, wild, aquarium, canal, dam, dock, pool, baker, barber, butcher, army, navy, police, change, difference, different, opposite, unequal, unlike, chase, follow, track, crumble, crumple, shorten, shrink, tighten, divorce, separate, split, outline, pattern, shape, exercise, practice, stretch, blister, burn, scab, sunburn, dark, shade, shadow, avalanche, earthquake, flood, hop, jump, leap, lobster, shell, shrimp, snail, starfish, collar, horseshoe, leash, saddle, cruel, mean, unkind, violent, alone, bother, upset, belief, doubt, hope, trust, fuel, gas, grease, oil, doorknob, handle, knob, dial, ladder, pedal, switch, trigger, guest, stranger, visitor, sled, sleigh, snowplow, name, title, nickname, law, regulation, rule, church, shrine, temple, open, shut, strong, weak, delicate, barn, shed, thing, object, sharp, dull, angle, diameter, radius, secret, private, grow, survive, giant, dwarf, tractor, wheelbarrow, free, liberty, obedient, author, speaker, writer, garbageman, janitor, custodian, station, airport, stomach, heart, sand, pebble, quit, stop, kick, stamp, average, sum, total, gorilla, monkey, become, seem, pioneer, caveman, citizen, star, celebrity, admit, tattle, record, recording, video, attention, interest, process, recipe, routine, belief, opinion, bashful, shy, dishonest, naughty, unfair, faucet, hose, sprinkler, cloud, fog, barefoot, naked, boss, leader, owner, babysitter, paperboy, astronaut, geography, scientist, guard, prisoner, slave, carpenter, plumber, judge, lawyer, maid, servant, forest, jungle, field, prairie, building, tower, office, shop, farm, ranch, pack, tape, tie, wrap, fail, succeed, luckily, unfortunately, magic, trick, blind, cold, deaf, reflect, shine, twinkle, measure, weigh, thermometer, yardstick, dry, overcast, sunny, ash, smoke, caffeine, helium, oxygen, guilt, shame, worry, grouch, grumpy, rude, amaze, excite, surprise, skill, talent, beginner, expert, promise, define, lazy, lucky, strict, holy, careful, sideways, afloat, waiter, mailman, cowboy, customer, secretary, pilot, desert, hospital, monument, audience, plant, force, germ, invisible, cloud, neat, crawl, stand, math, have to, event, vote, pipe, paint, scare, jealous, magnet, machine, camera"
    tier_one_words = tier_one_words.split(',')
    tier_one_words_list = []
    for word in tier_one_words:
        stripped_word = word.strip()
        tier_one_words_list.append(stripped_word)
    
    return tier_one_words_list

