# 

Powered by  [Jekyll.](http://jekyllrb.com)
Forked from [hyde](https://github.com/poole/hyde) and tweeked *a lot*.

Fell free to copy/modify/distribute!



## Initial tweeks to vanilla hyde:
1. Changed markdown parser from 'redcarpet' to 'kramdown'.
		Redcarpet wasn't rendering code and mathajax neatly
		Removed relative_permalinks: true  and 'highlighter: pygments.rb'
2. Added following plugins
	- [jemoji](https://github.com/jekyll/jemoji)
	- [jekyll-seo-tag](https://github.com/jekyll/jekyll-seo-tag)
	- [jekyll-sitemap](https://github.com/jekyll/jekyll-sitemap)
	
3. Styles
	1. Changed syntax highligher. Orignal hyde syntax css was to dull. So replaced. I don't have link to source/	
	2. Added [font-awesome](https://github.com/FortAwesome/Font-Awesome) stylesheets for rendering social media iconn
	3. Modified hyde.css and poole.css to adjusts sidebar fonts, paddings and width
	4. Replaced orignal favicon by mine.
	
4. Latex
	latex is not supported directly, hence add mathajax.html in _INCLUDES folder to render latex text
	
5. Analytics
	- Added google_analytics.html snippet in _INCLUDES folder
	- included same html in head
	
6. Posts
	- Deleted orignal posts
	
7. New:
	'jupyter' contains notebooks
	'images' contains site imgages refered from posts
	
8. Removed:
	CNMAME: As site is hosted on Gihtub, not required
	

## Tutorials/Links for modifying:
	[Markdown cheatsheet](https://www.markdownguide.org/cheat-sheet/)
	[Jekyll's](https://jekyllrb.com/)
	[Poole/hyde](http://hyde.getpoole.com/)
	

## Steps to add posts:
	1. Copy exising post from /_posts
	2. Erase everything except yaml info at the top
	3. Paste markdown content under yaml header
	4. Paste related images in /images/px folder where x stands for pot number
	5. Change images/assets references like this <img src="/images/px/filename.png">
	6. Save post with given filename format: YYYY-MM-DD-postname.markdown
	4. To refer jupytyer notebook in post, place jupyter notebook in /jupyterbooks folder in root
	5. Rename that jupyter notebook to post's filename. Like this: YYYY-MM-DD-postname.ipynb (Nothing else, just standardizes fromat for myself)
	
	
## Steps to add page:
	1. Just copy any old page from root folder. Rename it and change it's yaml content
	
	
## Steps to start jekylls locally
	1. Assuming Jekylls is installed properly from [here](https://jekyllrb.com/)
	2. Open CMD. Goto ostwalprasad.github.io directory. Pull one if you don't have.
	3. Type 'jekyll serve'. ( There's no gem file to run in this directory. Not sure why!) 
	4. goto localhost:4000
	5. Push changed to repo after changes are done.
	


## TODO:
Add disquis comments plugin
Site not rendering properly on medium resolution. Need to check
Google analytics tracking of subpages 

	
## License
Open sourced under the [MIT license](LICENSE.md).


