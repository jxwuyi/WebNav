import urllib2
import re
import myparser
import time

root = 'http://schools-wikipedia.org/wp/index/subject.htm'
#f = urllib2.urlopen(url)

out = open('content.txt','w')
log = open('logfile.txt','w')

tot_pages = 0
ptr = 0
que = [root]
vis = {root}
ps = myparser.MyParser()

start_time = time.time()
report_gap = 100

while (ptr < len(que)):
    url = que[ptr]
    ptr = ptr + 1
    log.write(url + '\n')
    result = ps.parse(url)
    if (result != None):
        title, text, links = result
	tot_pages += 1
        out.write(title + '\n')
        out.write(url + '\n')
        for tar in links:
            if ((tar != None) and (not tar in vis) and (len(tar) > 1)):
                vis.add(tar)
                que += [tar]
    if (ptr % 100 == 0):
        print ' >> %d / %d pages processed ....' % (tot_pages, ptr)
        elapsed = time.time() - start_time
        print '    Time Elapsed %f s' % elapsed

out.write('total number of pages = %d\n' % tot_pages)
print 'total page parsed = %d (among %d)\n' % (tot_pages, ptr)
