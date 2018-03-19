# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:27:31 2018

@author: zhaos
"""
import datetime as dt
import io
import re
import time
import numpy as np

class Header:
  """header of the EDF(+) file"""
  def __init__(self):
    self.version = None 
    self.demo=None
    self.recinfo=None
    self.epoch = None
    self.hdrbytes=None
    self.ndr = None
    self.dur = None
    self.ns = None
    self.other = None
    
    
class EDFEEG:
  """class that encapsulates the data of an EDF file"""
  header = None
  siginfo = None
  signals = None
  def samp_rate(self, chan_N):
    """returns the sampling rate of @chan_N"""
    return self.siginfo[chan_N].nsamp / self.header.dur
  def labels(self):
    """returns list of labels of signals"""
    return [i.label for i in self.siginfo]

def parseHdr(bb, parsestyle):
  """parses the header"""
  h = Header()
  h.version = int(bb[0:8].rstrip())
  if parsestyle == 'default':
      h.demo = parseptinfo(bb[8:88])
      h.recinfo = parserecinfo(bb[88:168])
  else:
      h.demo = ""
      h.recinfo = ""
      h.other = bb[8:168].decode()
  try:
    h.epoch = dt.datetime.strptime(bb[168:184].decode(),
          "%d.%m.%y%H.%M.%S")
  except ValueError:
    h.epoch = bb[168:184].decode()
  h.hdrbytes = int(bb[184:192])
  h.ndr = 20           #todo
  #h.ndr = int(bb[236:244])
  print (bb[236:244])
  h.dur = float(bb[244:252])
  h.ns = int(bb[252:256])
  return h


def parseptinfo(bb):
  """parses demographic information"""
  [mrn, sex, bday, name, *rest] = bb.split(b" ")
  try:
    dob = dt.datetime.strptime(bday.decode(), "%d-%b-%Y").date()
  except ValueError:
    dob = bday
  d = Demographics()
  d.sex = sex.decode()
  d.dob = dob.decode()
  d.mrn = mrn.decode()
  try:
    lname, gname = name.split(b"_")
    d.name = (lname.decode(), gname.decode())
  except ValueError:
    d.name = name.decode()
  return d


class Demographics:
  """demographic information"""
  def __init__(self):
    self.name = None
    self.sex = None
    self.dob = None
    self.mrn = None

def parserecinfo(bb):
  """parses recording information"""
  [stmark, stdate, eegnum, techcode, equipcode, *rest] = bb.split(b" ")
  try:
    recdate = dt.datetime.strptime(stdate.decode(), "%d-%b-%Y").date()
  except ValueError:
    recdate = stdate.decode()
  return {  "recdate" : recdate,
            "eegnum"  : eegnum.decode(),
            "techcode"  : techcode.decode(),
            "equip" : equipcode.decode() }


def getit(x, a, b):
  """utility function, returns x[a:a+b]"""
  return x[a:a+b]

class ChannelInfo:
  """metadata about each channel in an EDF(+) file"""
  label = ""
  trans_type = ""
  ph_dim = ""
  ph_min = None
  ph_max = None
  dig_min = None
  dig_max = None
  prefilt = ""
  nsamp = -1 # per data record

class field:
  """class representing a field in the EDF file"""

  def __init__(self, fieldlabel, offset, length, post):
    """takes a label, an offset into the header, a length, and a field data type"""
    self.lbl = fieldlabel
    self.off = offset
    self.length = length
    self.post = post
  
  def postprocess(self, bytesvalue):
    if self.post == 'float':
      return float(bytesvalue.rstrip())
    elif self.post == 'int':
      return int(bytesvalue.rstrip())
    elif self.post == "str":
      return bytesvalue.rstrip().decode()
    elif self.post == 'lstr':
      return bytesvalue.decode()



def parsesighdrs(bb, i):
  """parses the header information for each signal in the file"""
  jj = [ChannelInfo() for i in range(i)]

  offsets = [ 
    field('label', 0, 16, 'str'),
    field('trans_type', 16, 80, 'lstr'),
    field('ph_dim', (16+80), 8, 'str'),
    field('ph_min', (16+80+8), 8, 'float'),
    field('ph_max', (16+80+8+8), 8, 'float'),
    field('dig_min', (16+80+8+8+8), 8, 'int'),
    field('dig_max', (16+80+8+8+8+8), 8, 'int'),
    field('prefilt',  (16+80+8+8+8+8+8), 80, 'lstr'),
    field('nsamp',  (16+80+8+8+8+8+8+80), 8, 'int') ]
  k = 0
  for fld in offsets:
    for j in range(i):
      a = getit(bb, k+j*fld.length + fld.off, fld.length)
      setattr(jj[j], fld.lbl, fld.postprocess(a))
    k += (i-1)*fld.length
  return jj


def storeit(sig, off, i, k, n):
  """copies signal data into @sig object"""
  sig[i,off[i]:off[i]+n] = k

     # storeit(sigs, offsets, j, tx_by_sig(buffers[j], ss.siginfo, j), nsamps[j])


def transform(qty, dmin, dmax, phmin, phmax):
  """function to transform 2-byte integer to actual value"""
  qq = (qty-dmin)/float(dmax-dmin)
  return qq*(phmax-phmin)+phmin


def tx_by_sig(qty, siginfo, i):
  return transform(qty,   siginfo[i].dig_min, siginfo[i].dig_max,
                          siginfo[i].ph_min, siginfo[i].ph_max)


def parsesignals(bb, ss):
  """function to parse signal data"""
  ns = ss.header.ns

  nsamps = [ss.siginfo[k].nsamp for k in range(ns)]
  sigs = np.zeros( (ns, max(nsamps)*ss.header.ndr), dtype='<f8')
  #sigs = np.zeros( (ns, 175), dtype='<f8')
  drecsize = sum(nsamps)
  offsets = [0 for i in range(ns)]
  buffers = [np.zeros( nsamps[i], dtype='<i2') for i in range(ns)]

  #loop over drecs
  for i in range(ss.header.ndr):
    k = i*drecsize
    #loop over sig chunks in a drec
    for j in range(ns):
      m = k + nsamps[j]*2
      buffers[j] = np.frombuffer(bb[k:m], dtype='<i2')
      #storeit(sigs, offsets, j, buffers[j], nsamps[j])
      storeit(sigs, offsets, j, tx_by_sig(buffers[j], ss.siginfo, j), nsamps[j])
      offsets[j] += nsamps[j]
      k = m
  return sigs


def main(file):
    eegrawf = io.open(file, 'rb')
    beeg = io.BufferedReader(eegrawf)

    eegstuff = EDFEEG()
    eegstuff.header = parseHdr(beeg.read(256), "default")
    eegstuff.siginfo = parsesighdrs(
        beeg.read(eegstuff.header.hdrbytes-256),
        eegstuff.header.ns)

    beeg.seek(eegstuff.header.hdrbytes)

    eegstuff.signals = parsesignals(beeg.read(-1), eegstuff)
    beeg.close()
    return eegstuff.signals

#print (main('test.edf'))





'''
time.sleep(2)
eegrawf = io.open('123.edf', 'rb')
beeg = io.BufferedReader(eegrawf)

eegstuff = EDFEEG()
eegstuff.header = parseHdr(beeg.read(256), "default")
eegstuff.siginfo = parsesighdrs(
        beeg.read(eegstuff.header.hdrbytes-256),
        eegstuff.header.ns)

beeg.seek(eegstuff.header.hdrbytes)
eegstuff.signals = parsesignals(beeg.read(-1), eegstuff)
print (eegstuff.signals.shape)
beeg.close()
'''