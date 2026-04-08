from compress import compress_with_gpt
from chunk import chunk_text

input_text = """Water
The effects of climate change on our water resources can have a big impact on our world and our lives. Patterns of where, when, and how much precipitation falls are changing as temperatures rise. Some areas are experiencing heavier rain events while others are having more droughts.
Flooding is an increasing issue as our climate is changing. Compared to the beginning of the 20th century,  precipitation events are stronger, heavier, and more frequent across most of the United States.
Drought is also becoming more common, especially in the Western United States. We are using more water during hot weather, especially for agriculture. Much like we sweat more when it is hot out,  hot weather causes plants to lose, or transpire, more water. Then, farmers must give their crops more water.
Snowpack is an important source of fresh water for many people. As the snow melts, fresh water becomes available for use. Snowmelt is particularly important in regions like the Western United States where there is not much precipitation in warmer months. But as temperatures warm, there is less snow and snow begins to melt earlier in the year. This means that snowpack is less likely to be a reliable source of water. 
Our food supply depends on climate and weather conditions. Higher temperatures, drought and water stress, diseases, and weather extremes create challenges for farmers and ranchers. Farmers, ranchers, and researchers can address some of these challenges by adapting their methods or creating and using new technology. But, some changes will be difficult to manage, like human and livestock health. Farmworkers can suffer from heat-related health issues, like exhaustion, heatstroke, and heart attacks. Heat can also harm livestock.")
"""
chunks = chunk_text(input_text)

compressed = compress_with_gpt(chunks[0])
print("Original tokens:", len(chunks[0]))
print("Compressed tokens:", len(compressed.split()))
ratio = len(chunks[0]) / len(compressed.split())
print(f"Compression ratio: {ratio:.1f}x")