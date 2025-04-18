import requests
from bs4 import BeautifulSoup

from ebooklib import epub, ITEM_DOCUMENT


#Getting Chapter contents
url = 'https://novelbin.com/b/the-extras-academy-survival-guide/chapter-'

ch_num = 68
end_ch = 70

source = requests.get(url+str(ch_num)).text
soup = BeautifulSoup(source, 'lxml')

ch_content = soup.find('div', class_="chr-c" ).text

##Removing Chapter title from chapter content
ch_content = ch_content.split('\n')

ch_title = ''
i=0

while ch_title == '':
    ch_title = ch_content[i]
    i+=1

ch_content = '\n'.join(ch_content[i:])

title = soup.title.text

##Assigning Chapter number and name to ch_title
if 'Chapter' in title:
    for i in range(len(title)-1):
        if title[i].isnumeric() and not title[i+1].isnumeric():
            
            title = title[:i+1].split('#')
            ch_title = title[-1] +' '+ str(ch_title)
            title = title[0]
            
            break


if ch_num < end_ch:
    ch_num+=1   


#Saving to Epub

novel = epub.EpubBook()


#adding cover
with open("the-extras-academy-survival-guide.jpg", "rb") as img:
    novel.set_cover("the-extras-academy-survival-guide.jpg", img.read())


#Setting title and language
novel.set_title(title)

novel.set_language("en")

#Adding content
novel_chapter = epub.EpubHtml(title=ch_title,
                              file_name=f"chapter{ch_num}.xhtml",
                              lang='en')
novel_chapter.set_content(f"""<html><h1>Chapter {ch_num}</h1><body>{content}</body></html>""")


# In[158]:


novel.add_item(novel_chapter)


# In[159]:


# Add navigation and define spine
novel.add_item(epub.EpubNav())
novel.add_item(epub.EpubNcx())

novel.spine = ["nav", novel_chapter]


# In[160]:


#Preview

for item in novel.get_items():
    if item.get_type() == ITEM_DOCUMENT:
        try:
            print(f"Title: {item.get_name()}")
            
            print('Done')
        except:
            continue
    if 'Chapter' in item.get_name():
        print('1')
        print(item.content)


# In[161]:


# Write the EPUB file
epub.write_epub(f"{novel_title}.epub", novel)
print("EPUB created successfully!")


# In[ ]:





# In[ ]:




