import sys
import os
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect, GetErGFingerprint

file_in  = sys.argv[1]
file_name = os.path.splitext(file_in)[0]
file_out = file_name+"_descr.tsv"
ms = [x for x in  Chem.SDMolSupplier(file_in) if x is not None]
ms_write = open(file_out, 'w')

names=[x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)

ms_write.write("ID")

for d in names:
  ms_write.write("\t")
  ms_write.write(str(d)) 

for d in range(166):
  ms_write.write("\t")
  ms_write.write("MACCS_" + str(d))

for e in range(1024):
  ms_write.write("\t")
  ms_write.write("MFP_" + str(e))

ms_write.write("\n") 

for i in range(len(ms)):
  ms_write.write(str(i))
  descrs = calc.CalcDescriptors(ms[i])
  for x in range(len(names)):
    ms_write.write("\t")
    ms_write.write(str(descrs[x]))

  maccs = MACCSkeys.GenMACCSKeys(ms[i])
  for x in range(len(maccs)):
    ms_write.write("\t")
    ms_write.write(str(maccs[x])) 

  morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(ms[i], useChirality=True, radius=2, nBits=1024)
  for x in range(len(morgan)):
    ms_write.write("\t")
    ms_write.write(str(morgan[x])) 

  ms_write.write("\n")
    

ms_write.close
