# (c) Tom Swinburne 2017 tomswinburne@gmail.com
import numpy as np
import re

class eam_info:
  def __init__(self,_pot_file,pair_style="eam/fs"):
    lc=0
    self.ele=[]
    if pair_style == "eam/fs":
        for line in open(_pot_file):
          lc+=1
          #print lc,line,re.sub("\t"," ",line).strip().split(" ")[1:],"\n\n\n"
          if lc == 4:
            for _ele in re.sub("\t"," ",line).strip().split(" ")[1:]:
              #print _ele
              if _ele != "":
                self.ele.append(_ele)
          if lc == 5:
            self.cutoff=np.fromstring(line.strip(),sep=' ')[-1]
          if lc == 6:
            self.mass=np.fromstring(line.strip()[:-3],sep=' ')[-2]
            self.lattice=np.fromstring(line.strip()[:-3],sep=' ')[-1]
            self.crystal=line.strip()[-3:].lower()
          if lc > 6:
            break
    elif pair_style == "meam/spline":
        print ("meam")
        # a bad hack: '#', 'Mo', '42', '95.94', '3.1674', 'bcc', '6.0'
        for _line in open(_pot_file):
            line = _line.strip().split(' ')
            self.ele.append(line[1])
            self.cutoff=float(line[6])
            self.mass=float(line[3])
            self.lattice=float(line[4])
            self.crystal=line[5]
            break
