# RTFM
 
Answers questions about Git from the [Pro Git](https://github.com/progit/progit2) book. The book is chunked and embedded into Redis. Each question pulls the nearest chunks by vector search and Gemini answers from those chunks only, citing the file each fact came from. When the chunks don't cover a question, it replies with a fixed refusal sentence.
 
Built while working through the "Build Your Own RTFM For Me Agent" coding challenge.



## Licence
 
Pro Git is written by Scott Chacon and Ben Straub and licensed [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). The book is downloaded at setup and not included in this repository.