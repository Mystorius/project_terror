import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import nltk


# from wordcloud import WordCloud, STOPWORDS
from PIL import Image

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

file = 'data/data_clean.csv'
df = pd.read_csv(file, sep=";", header=0, encoding="ISO-8859-1", low_memory=False)
df['casualties'] = df['nkill'] + df['nwound']
df['weaptype1_txt'] = np.where(
    df['weaptype1_txt'] == 'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)', 'Vehicle',
    df['weaptype1_txt'])


def plots():
    # # attacks per year
    # plt.subplots(figsize=(15, 6))
    # sns.countplot(x='iyear', data=df, palette='inferno')
    # plt.xticks(rotation=90)
    # plt.title('Total attacks per year')
    # plt.show()
    #
    # # casualties per year
    # count_casul = df.groupby('iyear')['casualties'].sum().to_frame()
    # plt.subplots(figsize=(15, 6))
    # sns.barplot(x=count_casul.index, y='casualties', data=count_casul, palette='inferno')
    # plt.xticks(rotation=90)
    # plt.title('Casualties per year')
    # plt.show()
    #
    # # attacks methods
    # plt.subplots(figsize=(15, 6))
    # sns.countplot('attacktype1_txt', data=df, palette='inferno', order=df['attacktype1_txt'].value_counts().index)
    # plt.xticks(rotation=45)
    # plt.title('Attacking Types by Terrorists')
    # plt.show()
    #
    # # weapons used
    # plt.subplots(figsize=(15, 6))
    # sns.countplot('weaptype1_txt', data=df, palette='inferno', order=df['weaptype1_txt'].value_counts().index)
    # plt.xticks(rotation=45)
    # plt.title('Weapon types by Terrorists')
    # plt.show()

    # # attack type vs casualties
    # casulties_per_type = df.groupby('attacktype1_txt')['casualties'].sum().to_frame()
    # plt.subplots(figsize=(15, 6))
    # sns.barplot(x=casulties_per_type.index, y='casualties', data=casulties_per_type, palette='inferno')
    # plt.xticks(rotation=45)
    # plt.title('Casualties per attack type')
    # plt.show()

    # lethality per attack type
    casulties_per_type = df.groupby('attacktype1_txt')['casualties'].sum().to_frame()
    casulties_per_type.drop(['Unknown'], inplace=True)
    count_per_type = df['attacktype1_txt'].value_counts()
    count_per_type = pd.DataFrame({'index': count_per_type.index, 'count': count_per_type.values})
    count_per_type.index = count_per_type['index']

    for index, row in casulties_per_type.iterrows():
        for index_, row_ in count_per_type.iterrows():
            if index == index_:
                ratio = row.values / row_[1]
                casulties_per_type.loc[index] = ratio
    plt.subplots(figsize=(15, 8))
    sns.barplot(y=casulties_per_type.index, x='casualties', data=casulties_per_type, palette='inferno')
    # plt.xticks(rotation=45)
    plt.xlabel('lethality ratio')
    plt.ylabel('Attack method')
    # plt.title('Casualties ratio per attack type')
    plt.savefig('images/lethality.png')
    plt.show()

plots()


def plots_grouped():
    df_group = df.loc[(df['gname'] == 'Taliban') | (df['gname'] == 'Islamic State of Iraq and the Levant (ISIL)') | (df['gname'] == 'Boko Haram')]

    # # attacks per year
    # plt.subplots(figsize=(15, 6))
    # sns.countplot(x='iyear', hue='gname', data=df_group, palette='inferno')
    # plt.xticks(rotation=90)
    # plt.title('Total attacks per year')
    # plt.show()

    # casualties per year
    count_casul = df_group.groupby(['iyear', 'gname'])['casualties'].sum().to_frame()
    count_casul_total = df_group.groupby(['gname'])['casualties'].sum().to_frame()
    print(count_casul_total)

    plt.subplots(figsize=(15,   8))
    sns.barplot(x=count_casul.index.get_level_values('iyear'), hue=count_casul.index.get_level_values('gname'), y='casualties', data=count_casul, palette='inferno')
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Number of casualties')
    # plt.title('Casualties per year')
    plt.savefig('images/casualties_per_year.png')
    plt.show()

    # attacks methods
    plt.subplots(figsize=(15, 8))
    sns.countplot(y='attacktype1_txt', hue='gname', data=df_group, palette='inferno', order=df['attacktype1_txt'].value_counts().index)
    # plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    # plt.title('Attacking Types by Terrorists')
    plt.xlabel('Total number of incidents')
    plt.ylabel('Attack method')
    plt.savefig('images/attack_methods')
    plt.show()

    # # weapons used
    # plt.subplots(figsize=(15, 6))
    # sns.countplot('weaptype1_txt', hue='gname', data=df_group, palette='inferno', order=df['weaptype1_txt'].value_counts().index)
    # plt.xticks(rotation=45)
    # plt.title('Attacking Methods by Terrorists')
    # plt.show()
    #
    # # attack type vs casualties
    # count_per_type = df_group.groupby(['attacktype1_txt', 'gname'])['casualties'].sum().to_frame()
    # plt.subplots(figsize=(15, 6))
    # sns.barplot(x=count_per_type.index.get_level_values('attacktype1_txt'), hue=count_per_type.index.get_level_values('gname'), y='casualties', data=count_per_type, palette='inferno')
    # plt.xticks(rotation=45)
    # plt.title('Casualties per attack type')
    # plt.show()


plots_grouped()


def word_cloud():
    df_group = df.loc[(df['gname'] == 'Taliban') | (df['gname'] == 'Islamic State of Iraq and the Levant (ISIL)') | (df['gname'] == 'Boko Haram')]
    mask = np.array(Image.open("data/ak_47.jpg"))
    # mask =np.array(Image.open("data/terrorist.png"))

    df['motive'] = np.where(df['motive'] == 'The specific motive for the attack is unknown.', '', df['motive'])
    motive = df.loc[(df['targtype1_txt'] == 'Tourists'), 'motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(motive)
    stopwords = nltk.corpus.stopwords.words('english')
    [stopwords.append(w_) for w_ in ['unknown']]
    words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords)

    wordcloud = WordCloud(max_font_size=250, max_words=1000, stopwords=STOPWORDS, background_color='white', width=500,
                          height=500, margin=0, mask=mask).generate(" ".join(words_except_stop_dist))

    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
    wordcloud.to_file("img/cloud_test.png")

# word_cloud()
