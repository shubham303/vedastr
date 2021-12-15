if __name__ == "__main__":
	label = "क़ख़ग़ज़ड़ढ़फ़य़"
	char_similarity_map = {"क़": "क", "ख़": "ख", "ग़": "ग", "ज़": "ज", "ड़": "ड", "ढ़": "ढ", "फ़": "फ", "य़": "य", "ङ":"ड"}
	label = "".join(map(str, [char_similarity_map[str(ch)] for ch in list(label) if str(ch) in
	                          char_similarity_map]))
	print(list(label))
	
