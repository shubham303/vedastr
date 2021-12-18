m = "ऀ  ँ  ं  ः ़ ॅ े ै ॆ ् ॎ ॕ"
V = "ऄ ई ऊ ऍ  ऎ ऐ ऑ ऒ ओ औ"
CH = "अ आ उ ए इ ऌ क  ख  ग ऋ  घ  ङ  च  छ  ज  झ  ञ  ट  ठ  ड  ढ  ण  त  थ  द  ध  न  ऩ  प  फ  ब  भ  म  य  र  ऱ  ल  ळ  ऴ  व  " \
     "श  ष  " \
     "स  ह ॐ क़  ख़  ग़  ज़  ड़  ढ़  फ़  य़  ॠ  ॡ"
v = "ा  ि  ी  ु  ू  ृ  ॄ  ॉ  ॊ  ो  ौ  ॎ  ॏ ॑  ॒  ॓  ॔    ॖ  ॗ ॢ  ॣ"
symbols = "।  ॥  ०  १  २  ३  ४  ५  ६  ७  ८  ९ %  /  ?  :  ,  .  -"

def is_valid_label(label):

	state = 0
	for ch in list(label):
		if ch in symbols:
			state = 0
			continue
		
		if ch in CH:
			state = 2
			continue
		
		if ch in V:
			state = 1
			continue
		
		if state == 0:
			if ch in v or ch in m:
				print(ch)
				return False
		
		if state == 1:
			if ch in v:
				print(ch)
				return False
			
			if ch in m:
				state = 0
				continue
		
		if state == 2:
			if ch in v:
				state = 3
				continue
			
			if ch in m:
				state = 0
				continue
		
		if state == 3:
			if ch in m:
				state = 0
				continue
			
			if ch in v:
				return False
	return True


if __name__ == "__main__":
	label = "'इंगेजमेंंट'"
	print(list(label))
	char_similarity_map = {"क़": "क", "ख़": "ख", "ग़": "ग", "ज़": "ज", "ड़": "ड", "ढ़": "ढ", "फ़": "फ", "य़": "य", "ङ":"ड"}
	label = "".join(map(str, [char_similarity_map [str(ch)] if str(ch) in char_similarity_map else str(ch) for ch in
	                          list(label)]))
	print(len(list(label)))
	print(is_valid_label("'इंगेजमेंंट'"))\
	
	
	