# My Code for the Netflix Prize

I'm not aware of folks having published their code for the Netflix Prize. Here's mine.  
Under the team name "Hi!", I competed alone in college. I did it mostly for fun, and to learn modern machine learning techniques. It was an incredibly valuable, but strenuous, time. Well worth it on all fronts, though.   
I peaked out at #45 or so, and then dropped out to work on my senior thesis, and came in #145 or so.    
What I learned in the process was that smarter wasn't always better -- make an algorithm, and then scale it up, and then make a dozen tweaks to it, and then average all of the results together. That's how you climbed the leaderboard.   
  

Anyhoo, I haven't touched this code in awhile, but perhaps it'll be useful to folks interested in competitive data mining.  
Specifically, the lessons I learned:


  - Get the raw data into a saved and manageable format *fast*. The easier it is to load your data in and start mutating it, the better.
  - If doing simple pivots on your data is hard, and slows you down from visualizing whats in your data, spend time making data structures which make that easy.  
  - Generalize. Iterate. If you have a method you think will work, but it has a lot of knobs, and you don't know the best way to set those knobs, make it easy for you to try *every possible iteration*. There is often not a good way to figure out what the _best_ approach is. You will have to try many of them in order to build up an intuition. Specifically, that means (for me) a pluggable architecture. If there's ten ways to try a particular step, make sure you write your overarching algorithm so that it takes a function that you can pass to it, as opposed to having a method hardwired in the code. That way, you can hotswap all your ideas.  
  - Speed is a feature. Of course you make sure it works first. But your goal is to see if something works. If an algorithm takes a day to run, but you can spend five hours making it run in 1/3 of a day, do it. You'll be running it over and over again, and you'll learn more if you can iterate. 


As for the technical nitty-gritty, everything that's speed sensitive is written in Cython, which was the best balance of speed and convenience in 2009. If I were to do it al again, I would use (Numba)[http://github.com/numba/numba].  

The original data is gone, I believe, but I might have it stored somewhere. I'll look for that.  