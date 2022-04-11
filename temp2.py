import os

root = "/media/shubham/One Touch/Indic_OCR/recognition_dataset/"
#root = "/home/shubham/Documents/7/"
files = [x[0] for x in os.walk(root)]
print(files)


for file in files:
	import lmdb
	try:
		env = lmdb.open(
					file,
					max_readers=32,
					readonly=False,
					lock=False,
					readahead=False,
					meminit=False)
					
		cache={}
		i=0
	
		with env.begin(write=False) as txn:
			n_samples = int(txn.get('num-samples'.encode()))
			for index in range(n_samples-1,n_samples):
				idx = index + 1  # lmdb starts with 1
				label_key = 'label-%09d'.encode() % idx
				try:
					label = txn.get(label_key).decode('utf-8')
				except:
					print("exception occurred {}", index , n_samples )
					break
				i = index
		
		i+=1
		#txn = env.begin(write=True)
		#txn.put('num-samples'.encode() , str(i).encode())
		#txn.commit()
		env.close()
		
	except Exception:
		print("")
		