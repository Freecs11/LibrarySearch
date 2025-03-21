# Django Project Commands & Guidelines & project description
project is in LibrarySearch
## Description
Web/mobile application : A web/mobile application is different from a web page by their user interaction : a page
is reduced to be uniquely static. User interaction can be explicit : the application changes in response to an action that
the user has just performed. In the case of a side effect, or a consequence of older user actions, the interaction is called
an implicit UI.
Search engine for a library : In this assignment, we refer to library any database containing sufficiently many
text documents. One such example is given at The Gutenberg Project, where documents are stored in various formats,
including ASCII text format. Much as with The Gutenberg Project’s database, a library can store tens of thousands of
text documents. It is humanly challenging to manually search for the right document when answering to questions such
as What are all the books telling the story of King Sargon ? Has King Sargon ever came to Saigon harbor ? As a tourist,
what can we expect in actual Saigon district ?, and so on.
A search engine in such a library is a web/mobile application whose primary feature is to point its users to the right
document, according to a search by keyword. Another feature could be to point the users to a recommended document
following their search history.
Content analysis and performance analysis : The content relevance of a search engine is a subjective score resulting
from a user test campaign. It could be simplified as rule #12 of The Joel Test. Contrasting to this human biased measure,
performance ratings of a search engine can be determined by stress tests and statistics over increasingly large datasets.
1 Statement of final assignment – PRIMARY CHOICE
WARNING : It is mandatory to mention “Final project – PRIMARY CHOICE” on the cover page of the project
report in the final package.
The project consists in developing a web/mobile application offering a search engine for a certain library of text
documents. The first requirement is about the data layer : one need to collect sufficiently many text documents, either by
hard-storage on disk, or pointers to contents on distant servers such as at The Gutenberg Project’s. The minimum size of
the library must be 1664 books. The minimum size of each book must be 104
(ten to the four; ten thousand) words.
Next, it is required to build the server logic and the client views offering the main features of a search engine. Here,
each project team can determine its own userstories, however, they must include :
— Explicit feature “Search” : Search documents by keyword. On user input a string S, the application returns a list
of text documents whose index table contains S.
— Explicit feature “Advanced search” : Search documents by RegEx. On user input a string RegEx, the application
returns : either a list of text documents whose index table contains a string S matching RegEx as regular expression (refer to Lecture 1 of UE DAAR for a formal definition of regular expressions); or a list of text documents
containing a string S matching RegEx as regular expression (Warning : this option may cause the application to
slow down considerably).
— Implicit feature of ranking : Ordering the presentation of the documents returned by above features. In response
to a search or an advanced search, the web/mobile application returns the list of documents ordered by relevance,
according to some mathematical definition of ranking : by decreasing number of occurrences of the keyword/regEx
in the document, by decreasing centrality ranking of Jaccard graph (refer to the ending slides of Lecture 8 for a
15 septembre 2024
DAAR Final assignment. PRIMARY CHOICE – page 2/3
formal definition of centrality ranking and the Jaccard distance/graph). It is required to implement at least one of
the following centrality rankings : closeness, betweenness, or pagerank. It is also required in the final report of the
project to : recall the definition of the centrality ranking measure in use; the implemented algorithm computing
this measure; as well as examples of the computation on well chosen samples of the > 1664 books in your project
team’s database.
— Implicit feature of recommendation : Suggestion of documents with a content similar to the last search. Along
with the response to a search request, the web/mobile application also returns : either a list of documents which
are vertices of the Jaccard graph (cf. Lecture 8) in the neighbourhood of the highest ranked documents matching
the search request (according to the above feature of ranking); or a list of most “clicked” documents by other users
when doing a similar search request.
The project is essentially the construction of a web/mobile application. It is important to include a demo version
during the oral presentation or the video pitch of the project (see description of oral presentation and video pitch below).
It is also very important that the application can be run on various client machines during the demo (laptops, smartphones
with iOS and/or Android, etc).

## Commands
- Run server: `python manage.py runserver`
- Create migrations: `python manage.py makemigrations`
- Apply migrations: `python manage.py migrate`
- Run all tests: `python manage.py test`
- Run app tests: `python manage.py test myImageBank`
- Run specific test: `python manage.py test myImageBank.tests.TestClassName.test_method_name`
- Django shell: `python manage.py shell`

## Code Style Guidelines
- Follow PEP 8 conventions for Python code
- Use 4 spaces for indentation (no tabs)
- Max line length: 79 characters
- Class names: CamelCase (e.g., `ImageModel`)
- Function/method names: snake_case (e.g., `get_image_url`)
- Variable names: snake_case (e.g., `image_list`)
- Use docstrings for classes and functions
- Import order: standard library, third-party, local application
- Configuration values should be placed in the respective app's config.py
- Use class-based views for REST endpoints
- Error handling: use try/except blocks with specific exception types