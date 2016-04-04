import urllib2
import re
import myparser
import time

root = 'http://schools-wikipedia.org/wp/index/subject.htm'
#f = urllib2.urlopen(url)

out = open('content.txt','w')

ptr = 0
que = [root]
vis = {root}
ps = myparser.MyParser()

start_time = time.time()
report_gap = 100

while (ptr < len(que)):
    url = que[ptr]
    ptr = ptr + 1
    result = ps.parse(url)
    if (result):
        title, text, links = result
        out.write(title + '\n')
        out.write(url + '\n')
        for tar in links:
            if (not tar in vis):
                vis.add(tar)
                que += [tar]
    if (ptr % 100 == 0):
        print ' >> %d pages processed ....' % ptr
        elapsed = time.time() - start_time
        print '    Time Elapsed %f s' % elapsed

out.write('total number of pages = ' + ptr)
