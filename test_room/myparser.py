import urllib2
import re
import myparameters as prm

class MyParser():
    
    def __init__(self):
        self.page_dat = {}
        f = open(prm.content_path,'r')
        p = 0
        last = ''
        for line in f:
            p = p + 1
            if (p % 2 == 0):
                now = line.replace('\n','')
                self.page_dat[now] = last
            else:
                last = line.replace('\n','')
        f.close()

    def gen_full_url(self, cur, target):
        if (not target.startswith('../')):
            return None
        if ('/images' in target):
            return None

        pos = target.find('#')
        if (pos > -1):
            target = target[:pos]
        
        #remove html
        pos = cur.rfind('/');
        cur = cur[:pos]
        while True:
            if (target.startswith('../')):
                pos = cur.rfind('/')
                cur = cur[:pos]
                target = target[3:]
            else:
                now = cur + '/' + target
                if (now in self.page_dat):
                    return now
                else:
                    return None
        return None
    
    def extract_links(self, links, text, cur):
        terms = ''
        curr = re.findall('href=[\'"]?([^\'" >]+)', text)
        for url in curr:  #should only find one url
            links.add(self.gen_full_url(cur, url))
            lp = text.find(url)
            rp = text.find('</a>')
            t = text[lp+len(url)+2:rp]
            terms += ' ' + t.lower()
        return terms

    def filter_links(self, links, text, cur):
        curr = re.findall('href=[\'"]?([^\'" >]+)', text)
        for url in curr:
            links.add(self.gen_full_url(cur, url))
            text = text.replace(' href="'+url+'"','')
        curr = re.findall('title=[\'"]?([^\'">]+)', text)
        for ttl in curr:
            text = text.replace(' title="'+ttl+'"','')
        curr = re.findall('class=[\'"]?([^\'">]+)', text)
        for cls in curr:
            text = text.replace(' class="'+cls+'"','')
        text = text.replace('<a>','').replace('</a>','').lower()
        return text

    def parse(self, url):
        f = urllib2.urlopen(url)
	if (not 'schools-wikipedia.org' in f.geturl()):
	    return None

        text = ''
        links = set()
        toRem = {'<!--del_lnk-->','<b>','</b>','<i>','</i>'}
        for line in f:
            if ('SOS Children' in line):
                continue #charity content
            
            if ('<title>' in line):
                title = line.strip().replace('<title>','').replace('</title>','').lower()

            if ('class="categoryname"' in line):
                # category page [a hack]
                text += ' ' + self.extract_links(links, line, url)

            if ('<td style="padding-right:2em"><a ' in line):
                # category terms [a hack]
                text += ' ' + self.extract_links(links, line, url)

            if ('<p>' in line):
                # process text
                lp = line.find('<p>')
                rp = line.find('</p>')
                cur = line[lp+3:rp] # get text
                for pat in toRem:
                    cur = cur.replace(pat,'') # remove tags
                text += ' ' + self.filter_links(links, cur, url)

        text = re.sub(r'\&lt\;ref.*?\&lt\;\/ref\&gt\;', '', text, flags=re.DOTALL)
        text = re.sub(r'\&lt\;ref.*?\/\&gt\;', '', text, flags=re.DOTALL)
        text = text.replace('\n','')
        return title, text, links
