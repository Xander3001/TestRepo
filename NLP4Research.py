# Databricks notebook source
##DOCUMENT CATEGORIZATION##

##WHY? 

# spam filtering, a process which tries to discern E-mail spam messages from legitimate emails
# email routing, sending an email sent to a general address to a specific address or mailbox depending on topic
# language identification, automatically determining the language of a text
# genre classification, automatically determining the genre of a text
# readability assessment, automatically determining the degree of readability of a text, either to find suitable materials for different age groups or reader types or as part of a larger text simplification system
# sentiment analysis, determining the attitude of a speaker or a writer with respect to some topic or the overall contextual polarity of a document.
### (health-related classification using social media in public health surveillance) 
# article triage, selecting articles that are relevant for manual literature curation, for example as is being done as the first step to generate manually curated annotation databases in biology
# Link university reserachers with other uni researchesr and company researchers based on their research output abstract, amount of projects done and amount of funds received.


##APPROACH: 

# Content-based classification is classification in which the weight given to particular subjects in a document determines the class to which the document is assigned.
# For example,  at least 20% of the content of a DOCUMENT should be about the class to which the book is assigned.


# COMMAND ----------

# from rasa_nlu.training_data import load_data
# from rasa_nlu.config import RasaNLUModelConfig 
# from rasa_nlu.model import Trainer
# from rasa_nlu import config

import databricks.koalas as ks
from spacy import displacy
from collections import Counter
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Usual imports
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import concurrent.futures
import time
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import warnings
warnings.filterwarnings('ignore')



# # Plotly based imports for visualization
# from plotly import tools
# import plotly.plotly as py
# from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
# import plotly.graph_objs as go
# import plotly.figure_factory as ff

###
import json
import sys
import os
import glob
import matplotlib
import re
import scipy
from pyspark.sql.types import *
from multiprocessing import Process, freeze_support

# gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import datapath
from gensim import corpora, models

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('stopwords')
nltk.download('wordnet')



import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
%matplotlib inline

# COMMAND ----------

# DBTITLE 1,Blob Connection
spark.conf.set(
  "fs.azure.account.key.nlpstorage01.blob.core.windows.net",
  "WsmeZi7Zg+reOosVE+tQ2nHw+JsUyKPM1XNjFl6ttfYIHpdQWhq6/x+E2V9dula5C1worY+umUAcRUYzCANk5w==")

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Load Data from blob
df1 = spark.read.csv("wasbs://nlpstore@nlpstorage01.blob.core.windows.net/wellcome-grants-awarded-2005-2018.csv", header='true')
df2 = spark.read.csv("wasbs://nlpstore@nlpstorage01.blob.core.windows.net/20200201_Innovate_UK_Funded_Projects.csv", header='true') ##fix csv file
#txtdf = spark.read.csv("wasbs://nlpstore@nlpstorage01.blob.core.windows.net/highrisktxt.txt", header=None)


display(txtdf)

# COMMAND ----------


ldaDf = df1.select("Description")


# COMMAND ----------

# DBTITLE 1,Spark DF to Pandas DF
pdf = ldaDf.select("*").toPandas()
ksdf = ks.from_pandas(pdf)
textdata = txtdf.select("*").toPandas()

# COMMAND ----------

ab = pdf.replace('Not available', 0, regex=True)
abst = ab.loc[~(ab==0).all(axis=1)]
absty = abst.replace('Summary not available', 0, regex=True)
cleandf = absty.loc[~(absty==0).all(axis=1)]
tr = cleandf.replace('Summary not avaialble', 0, regex=True)
bestdf = tr.loc[~(tr==0).all(axis=1)]

bestdf
#13912

# COMMAND ----------

print(len(textdata))

# COMMAND ----------

# DBTITLE 1,Install Spacy model 
# MAGIC %sh
# MAGIC python -m spacy download en_core_web_lg

# COMMAND ----------

# DBTITLE 1,Import & Define NLP model
import en_core_web_lg
nlp = en_core_web_lg.load()

# COMMAND ----------

# DBTITLE 1,Define Text and fit NLP model
doc = nlp(ksdf['Description'][0])
doc2 = nlp(ksdf['Description'][7])



# COMMAND ----------

la = "\n\n\n[bookmark: BMK_Cover]\n\n\n\tDated\t\nsimplyhealth group limited\nAND\n ADATIS GROUP LIMITED\n\nFramework agreement for the provision of services\n\n\n[image: ]\n\nVersion 01.09.2019\n\nSimplyhealth Group Limited, registered in England and Wales with company number 05445654, with registered office at Hambleden House, Waterloo Court, Andover, Hampshire, SP10 1LQ\nContents\nClause\tPage\n1\tDefinitions and interpretation\t1\n2\tCommencement and term\t4\n3\tSupplier obligations\t5\n4\tSimplyhealth obligations\t5\n5\tPurchase Orders\t6\n6\tPerformance of the Services\t6\n7\tChange control procedure\t7\n8\tWarranty\t8\n9\tPrice\t9\n10\tPayment\t9\n11\tData protection\t10\n12\tInsurance\t10\n13\tIntellectual property rights\t10\n14\tLimitation of liability\t11\n15\tIndemnity\t12\n16\tTermination\t13\n17\tPersonnel and TUPE\t15\n18\tNon-solicitation\t17\n19\tConfidential information\t17\n20\tAnti-bribery\t17\n21\tModern slavery\t18\n22\tAnti-tax evasion facilitation\t19\n23\tAudits and investigations\t19\n24\tDispute resolution\t20\n25\tForce majeure\t20\n26\tEntire agreement\t20\n27\tNotices\t21\n28\tAnnouncements\t22\n29\tFurther assurance\t22\n30\tVariation\t22\n31\tAssignment\t22\n32\tSet off\t23\n33\tNo partnership or agency\t23\n34\tEquitable relief\t23\n35\tSeverance\t23\n36\tWaiver\t23\n37\tCompliance with law\t24\n38\tConflicts within agreement\t24\n39\tCounterparts\t24\n40\tCosts and expenses\t24\n41\tThird party rights\t24\n42\tGoverning law\t25\n43\tJurisdiction\t25\nSchedule 1 \nServices\t26\nSchedule 2\nSimplyhealth IT Access Policy\t27\nSchedule 3\nData protection\t28\nPart 1 - Operative provisions\t28\nPart 2 - Data processing and security details\t33\nSchedule 4\nChange request\t35\n\n\n\n\n\n\n\n10-30868808-2\\326497-153\nVersion 01.09.2019\n\nSimplyhealth Group Limited, registered in England and Wales with company number 05445654, with registered office at Hambleden House, Waterloo Court, Andover, Hampshire, SP10 1LQ\t\nThis Agreement is made on \nBetween\n[bookmark: _e7c60873-ae4a-40db-b74d-732ff6a906dc]Simplyhealth Group Limited a company incorporated in England and Wales under number 05445654 whose registered office is at  Hambleden House, Waterloo Court, Andover, SP10 1LQ (Simplyhealth); and\n[bookmark: _3e9fc2ec-651d-4ad1-8900-42ae2bb8de7a]Adatis Group Limited a company incorporated in England and Wales under number 11788735 whose registered office is at Broadmeads House, Farnham Business Park, Farnham, Surrey, GU9 8QT (Supplier) \n(each of the Supplier and the Simplyhealth being a party and together the Supplier and Simplyhealth are the parties).\nWhereas\n[bookmark: _d37f2313-fe7a-449e-bbc2-03609d2066c5]Simplyhealth conducts the business of providing access to healthcare.\n[bookmark: _099604d3-a858-4853-a9dd-fe3e3bb28450]The Supplier conducts the business of supplying consultancy services to other businesses.\n[bookmark: _bd5c30e7-6d92-4037-95ac-e9861b0b2334]The parties have agreed that the Supplier shall supply services to Simplyhealth on the terms set out in this Agreement.\n[bookmark: _01655870-3fac-4019-b128-8e8a3679cca4]The parties contemplate that the Supplier shall supply services to Simplyhealth on a call-off basis when requested by Simplyhealth.\nIt is agreed\n[bookmark: _Toc256000000][bookmark: _Toc26196063][bookmark: _c1949172-e209-45dd-ac2e-f4e8f6574149]Definitions and interpretation\n[bookmark: _eddaea38-ca6f-463a-be0d-bb1fb9d8034c]In this Agreement:\nAffiliate means any entity that directly or indirectly Controls, is Controlled by, or is under common Control with, another entity;\n[bookmark: _Hlk26441088]Best Industry Practice means in relation to any undertaking and any circumstances, the highest degree of professionalism, skill, diligence, prudence and foresight which would be expected from an internationally recognised and market leading company engaged in the same type of activity under the same or similar circumstances and which is best in class;\tComment by Martin Philpott: See my comments in section 8.0\nBribery Laws means the Bribery Act 2010 and associated guidance published by the Secretary of State for Justice under the Bribery Act 2010 and all other applicable United Kingdom laws, legislation, statutory instruments and regulations in relation to bribery or corruption and any similar or equivalent laws in any other relevant jurisdiction;\nBusiness Day means a day other than a Saturday, Sunday or bank or public holiday in England;\nChange means any change to this Agreement including to any of the Services or to any of the Purchase Orders;\nChange Control Procedure means the process by which any Change is agreed as set out in clause 7;\nChange Request means a request submitted by a party to effect a Change, in the form of the template at Schedule 4;\nCommencement Date means the date of this Agreement; \nCompletion shall, in relation to each Purchase Order, have the meaning given to it in clause 6.3, and Completed and similar expressions shall be construed accordingly; \nConfidential Information has the meaning given in clause 19.1;\nControl has the meaning given in the Corporation Tax Act 2010, s 1124 and Controls and Controlled shall be interpreted accordingly;\nDeliverables means the goods ancillary to the supply of the Services, to be supplied by the Supplier to Simplyhealth;\nEmployee means any employee (as defined in the TUPE Regulations) of Simplyhealth who is assigned to providing services akin to the Services;\nForce Majeure has the meaning given in clause 25.1;\nIntellectual Property Rights means copyright, patents, rights in inventions, rights in confidential information, Know-how, trade secrets, trade marks, service marks, trade names, design rights, rights in get-up, database rights, rights in data, semi-conductor chip topography rights, mask works, utility models, domain names, rights in computer software and all similar rights of whatever nature and, in each case: (i) whether registered or not, (ii) including any applications to protect or register such rights, (iii) including all renewals and extensions of such rights or applications, (iv) whether vested, contingent or future and (v) wherever existing;\nKnow-how means inventions, discoveries, improvements, processes, formulae, techniques, specifications, technical information, methods, tests, reports, component lists, manuals, instructions, drawings and information relating to customers and suppliers (whether written or in any other form and whether confidential or not);\nLaw means\n[bookmark: _649816d1-6cea-43c0-9758-d283196f2147]any law, statute, regulation, by-law or subordinate legislation in force from time to time to which a party is subject and/or in any jurisdiction that the Services are provided to or in respect of;\n[bookmark: _1b0327e4-1b64-4325-bcbd-2a245eedb5e8]the common law and laws of equity as applicable to the parties from time to time;\n[bookmark: _a150745a-433b-4e87-9ce8-a06ee09c057b]any binding court order, judgment or decree;\n[bookmark: _afb69d55-c4ea-42f6-a493-7e777a4bca19]any applicable industry code, policy or standard; or\n[bookmark: _b71a76d6-1dd9-4dec-96e8-b507477791cf]any applicable direction, policy, rule or order that is binding on a party and that is made or given by any regulatory body having jurisdiction over a party or any of that party’s assets, resources or business;\nMilestone means an activity, process or outcome described in an Purchase Order relating to the Services to be provided under that Purchase Order;\nMilestone Date means the date set out in the Purchase Order by which the Milestone must have been achieved by the Supplier;\nModern Slavery Policy means Simplyhealth's anti-slavery and human trafficking policy in force and notified to the Supplier from time to time;\nPerformance Location means the location set out in an Purchase Order at which the Services shall be performed;\nPurchase Order has the meaning given in clause 5.1;\nPolicies and Procedures means any Simplyhealth policies and procedures as the same may be updated from time to time by Simplyhealth, the current versions of which can be provided on request. Policies and procedures includes the Simplyhealth IT Access Policy as set out in Schedule 2;\nRestricted Period means the Term and a period of 6 months after the Term; \nRestricted Person means any person employed or engaged by a party at any time during the Term who has or had material contact or dealings with the other party;\nServices means, as the context permits, (i) the services listed in Schedule 1 or (ii) the services supplied to Simplyhealth by the Supplier pursuant to a Purchase Order, together with the corresponding Deliverables (where the context permits);\nServices Commencement Date means the first date on which the Supplier provides the Services to Simplyhealth;\nServices Termination Date means the final date on which the Supplier provides the Services to Simplyhealth;\nSimplyhealth Materials means any material owned by Simplyhealth or its Affiliates relating to the Services (and any modifications to that material);\nSpecification means the description of the Services set out in a Statement of Work;\nStatement of Work means the detailed activities, timetable, dependencies and sequence of events which the Supplier shall perform, or procure the performance of, when delivering the Services agreed between the parties pursuant to clause 5.2 and forming part of a Purchase Order;\nSuccessor Supplier means any person taking responsibility for the provision of the Services following termination of this Agreement;\nSupplier Personnel means all employees, officers, staff, other workers, agents and consultants of the Supplier, its Affiliates and any of their subcontractors who are engaged in the performance of the Services from time to time;\nTerm has the meaning set out in Clause 2.1;\nTermination Assistance means all necessary assistance (which shall include knowledge transfer), as may be reasonably required by Simplyhealth to complete the transition of all or part of the Services from the Supplier to a third party designated by Simplyhealth or to Simplyhealth at Simplyhealth’s election and request;\nVAT means value added tax chargeable under the Value Added Tax Act 1994 and any other tax of any jurisdiction based on sales of goods such as sales taxes and any similar, replacement or additional tax.; and\n[bookmark: _d2fd18ce-630b-4dae-af09-979edfbf019e]In this Agreement:\n[bookmark: _dc33f874-5b9a-4c12-8a4e-3c74ad23e2af]a reference to this Agreement includes its schedules, appendices and annexes (if any);\n[bookmark: _6f2fe4de-002f-47f7-aacb-a32f940cfa56]a reference to a ‘party’ includes that party’s personal representatives, successors and permitted assigns;\n[bookmark: _cefc0aee-bdd1-4f9e-825f-13df4865ec52]a reference to a ‘person’ includes a natural person, corporate or unincorporated body (in each case whether or not having separate legal personality) and that person’s personal representatives, successors and permitted assigns;\n[bookmark: _c71e3e62-ead1-434c-b480-1a5761440c20]a reference to a gender includes each other gender;\n[bookmark: _188e9827-64bc-4625-9b4b-82e9ce5c10e1]words in the singular include the plural and vice versa;\n[bookmark: _cd1c720d-acb3-4776-9dea-1bfc7d6ca12b]any words that follow 'include', 'includes', 'including', ‘in particular’ or any similar words and expressions shall be construed as illustrative only and shall not limit the sense of any word, phrase, term, definition or description preceding those words;\n[bookmark: _229f6ad0-3b74-4d08-ae20-a238f191cc53]the table of contents, background section and any clause, schedule or other headings in this Agreement are included for convenience only and shall have no effect on the interpretation of this Agreement; and\n[bookmark: _eb74fa61-fa8c-424c-9c2c-4d3a64d96b62]a reference to legislation is a reference to that legislation as  amended, extended, re-enacted or consolidated from time to time except to the extent that any such amendment, extension or re-enactment would increase or alter the liability of a party under this Agreement.\n[bookmark: _Toc256000001][bookmark: _Toc26196064][bookmark: _ce15b54e-3fef-47a6-b5f2-e0368d5fd492]Commencement and term\n[bookmark: _f3c7bbb5-b037-4bc3-aafd-0cf2348798db]This Agreement commences on the Commencement Date and shall continue in force until 31st December 2020 whereupon it shall automatically terminate, unless terminated earlier by the parties pursuant to clause 6.5 (a), clause 16 or clause 25.3 (the Term).\n[bookmark: _Toc256000002][bookmark: _Toc26196065][bookmark: _a73a7d1c-a980-4067-8772-a2a60efca644]Supplier obligations\n[bookmark: _a4fe1439-9195-4e44-9350-d99b2dba50bc]During the Term, the Supplier agrees to supply, and Simplyhealth agrees to purchase, Services on the terms set out in this Agreement.\n[bookmark: _a4e37e7a-9b44-4acb-b298-b13b7e6e5f38]The Supplier shall, and shall procure that the Supplier Personnel shall at all times and in all respects:\n[bookmark: _f23ead2d-da6c-4d89-840e-3c500b9d39a4]perform the Services in accordance with the terms of this Agreement and each of the Purchase Orders;\n[bookmark: _15c95c40-3f90-4a63-8f41-63920fec3137]achieve the Milestones by the Milestone Dates set out in each Purchase Order;\n[bookmark: _8c29f544-fba9-406a-b33a-ccd9d7fe292c]comply with the Policies and Procedures;\n[bookmark: _3d437dcd-8243-463e-92e0-9438b7c1da38]comply with any additional or special responsibilities and obligations of the Supplier specified in each Purchase Order;\n[bookmark: _b12e24a3-53c8-414c-8824-8b51c156cd1b]co-operate with Simplyhealth in all matters arising under this Agreement or otherwise relating to the performance of the Services;\n[bookmark: _d45d6ffd-f1a6-478c-b6d1-4f1e62d792ef]use the Performance Location in an efficient manner and for the sole purpose of providing the Services;\n[bookmark: _e2ff4e0b-74e2-4b98-8359-25015aff831d]provide all information, documents, materials, data or other items necessary for the provision of the Services to Simplyhealth in a timely manner;\n[bookmark: _4872b454-14be-4c93-9771-47c7cbefc75b]inform Simplyhealth in a timely manner of any matters (including any health, safety or security requirements) which may affect the provision of the Services or the performance of any Purchase Order; \n[bookmark: _4f3e5dab-0b20-4ee1-90b7-3c1948112372]ensure that all tools, equipment, materials or other items used in the provision of the Services are suitable for the performance of the Services, in good condition and in good working order; and\n[bookmark: _59cd4838-d8e6-4ad8-8d9a-99316e96b9e2]obtain and maintain all necessary licences, permits and consents required to enable it to perform the Services and otherwise comply with its obligations under this Agreement.\n[bookmark: _aae685d6-74cc-4e74-9b3e-fd73c92841d0]The Supplier shall ensure that it has sufficient, suitable, experienced and appropriately qualified Supplier Personnel to perform this Agreement.\n[bookmark: _Toc256000003][bookmark: _Toc26196066][bookmark: _dc09a3ac-9166-48ea-9878-33f3e80d40b5]Simplyhealth obligations\n[bookmark: _1b38a699-5b92-4006-bea4-a7f381ac6622]To the extent reasonably necessary for the Supplier to perform its obligations under this Agreement, Simplyhealth shall provide or procure for the Supplier and/or Supplier Personnel:\n[bookmark: _e73b7271-c741-45d2-b528-57496f024c19]access to Simplyhealth Materials; and\n[bookmark: _a14015d0-8022-4ec2-a41f-e2b1e88f6c36]access to the Performance Location.\n[bookmark: _Toc256000004][bookmark: _Toc26196067][bookmark: _943cd74b-9e2e-4df9-8889-8b3348c207fc]Purchase Orders\n[bookmark: _46949bce-ac2d-42d7-a15e-458d882ca144]Simplyhealth may at any time provide the Supplier with a written order for Services, provided always that where the Services are Services of the type which require the parties to agree a Statement of Work then that Statement of Work shall first be agreed by the parties pursuant to clause 5.2 (a Purchase Order). \n[bookmark: _35a10cae-e82f-4f3b-80c9-5274efc53b4e]Where the Services required by Simplyhealth are Services of the type which require the parties to agree a Statement of Work, then:\n[bookmark: _f3dc6720-dab4-4736-b697-b57a5134866b]Simplyhealth shall submit a draft Purchase Order for such Services to the Supplier requesting the Supplier to submit a corresponding draft Statement of Work;\n[bookmark: _5dcdf87d-63d3-4191-b913-11e1766457fa]the Supplier shall, at its cost and expense, submit a draft Statement of Work to Simplyhealth within five (5) Business Days of the date of the draft Purchase Order;\n[bookmark: _b75e663b-cffc-487e-95b9-c9d6af4eecec]the Supplier shall, at its cost and expense, promptly provide all necessary advice, support and assistance as may be required by Simplyhealth from time to time in considering the draft Statement of Work;\n[bookmark: _1d7f07dc-13d9-4608-9327-713c4c72f773]the Supplier shall, at its cost and expense and promptly, update and amend the draft Statement of Work from time to time as necessary as a result of its interactions with Simplyhealth pursuant to clause 5.2 (c); and\n[bookmark: _bfc80efd-9a50-411f-b16b-3a285ed4a041]the Supplier and Simplyhealth shall sign the Statement of Work when it is agreed and the signed Statement of Work shall complete the draft Purchase Order.\n[bookmark: _1aa4de0f-6ae5-4c89-8a4c-a02b1c9690fc]Simplyhealth shall be entitled to amend or withdraw a Purchase Order by giving the Supplier notice in writing in relation to any Services where performance has not commenced.\n[bookmark: _74fbe4cf-62ff-49f1-9553-2ee44e6ca750]Each Purchase Order shall constitute a binding obligation on the Supplier to supply the Services in accordance with the terms of the Purchase Order and this Agreement.\n[bookmark: _5f581a0f-4e56-45a6-8511-8ec18193f02a]No variation to a Purchase Order shall be binding unless expressly agreed in writing and executed by a duly authorised signatory on behalf of Simplyhealth, or otherwise in accordance with the provisions of clause 8.\n[bookmark: _5fafb7b8-85c8-48f7-b7f3-32df695bff2c]No Purchase Orders shall be placed following the date on which notice is validly served pursuant to clauses 6.5 (a), 16 or 25.3, or the date on which the Agreement expires pursuant to clause 2.\n[bookmark: _b584c7c1-9526-443b-8f10-699e92a041ba]Each Purchase Order shall form part of and be interpreted in accordance with the provisions of this Agreement.\n[bookmark: _Toc256000005][bookmark: _Toc26196068][bookmark: _34676aa4-649f-4a8c-aff9-5acd3727ef2d]Performance of the Services\n[bookmark: _81527a2c-1e35-4b75-9b9c-b7f8ef60ddbc]Each Purchase Order shall specify the Performance Location and Simplyhealth shall make such premises available for the Supplier.\n[bookmark: _98ba5c2e-bfac-427c-81a0-e0f1bf69edab]The Supplier shall perform the Services in accordance with any commencement or end dates specified for performance and shall achieve the Milestones by the Milestone Dates, in each case as specified in the corresponding Purchase Order. Services which do not have specified commencement or end dates or Milestone Dates shall be performed by the Supplier as soon as possible but, in any event, within a reasonable period of time.\n[bookmark: _78c521d4-b6a9-443f-9fc6-ca8287fb4aab]Each Purchase Order shall be deemed to have been completed at such time as Simplyhealth is satisfied that the Services have been performed by the Supplier in full and in accordance with the terms of this Agreement and the terms of the corresponding Purchase Order (Completion).\n[bookmark: _bd4a1da4-aafd-4011-ac22-9d1f7c54e32c]Following performance of the Services, the Supplier shall provide a completion note to Simplyhealth stating:\n[bookmark: _060c1585-65b4-4e97-857b-519b236838ab]the date and reference number of the Purchase Order;\n[bookmark: _50e72590-b05f-4220-bd20-46c204f82563]a description of the Services performed;\n[bookmark: _a71b2504-7593-4faf-a7e7-e105cf99ded4]the categories, type and quantity of any Deliverables supplied; and\n[bookmark: _637ddcc1-baf7-4bea-8239-90fef4195c28]any further information identified as being required in the corresponding Purchase Order.\n[bookmark: _1ed5711e-d1a1-4de3-9cb2-174fccb9733f]Time of performance shall be of the essence. Subject to the provisions of clause 6.6, if in relation to a Purchase Order, the Supplier fails to comply with the provisions of clause 6.2, then Simplyhealth may:\n[bookmark: _1bb35847-955c-4088-a92f-7a6241a4f091]refuse to accept any subsequent attempts to perform the Purchase Order under the Agreement and terminate the Purchase Order immediately by written notice to the Supplier;\n[bookmark: _afd6ce9a-5d1d-4cdf-8cd4-919749db136a]procure services similar to the Services identified in the Purchase Order from an alternative supplier; and\n[bookmark: _e7626b3e-748f-451f-b7f1-44f1332eadd9]recover from the Supplier all losses, damages, costs and expenses incurred by Simplyhealth arising from the Supplier’s default.\n[bookmark: _6686b4bf-a4c0-45ef-9d53-801f82b70f16]The Supplier shall not be liable for any failure to comply with the provisions of clause 6.2 to the extent such failure is caused by:\n[bookmark: _671e13fd-8981-4921-963e-8278fa48badc]Simplyhealth's failure to: (i) make the Performance Location available or (ii) prepare the Performance Location as reasonably necessary; or\n[bookmark: _d188f23e-794a-49bf-8b50-9c9297076ec6]an event of Force Majeure.\n[bookmark: _Toc256000007][bookmark: _Toc26196069][bookmark: _c5a6572e-2758-4b3a-bd9c-28770e4fc808]Change control procedure\n[bookmark: _1c098aef-ffc0-44e5-8048-91aa1775ad3a]Where Simplyhealth or the Supplier sees a need to change this Agreement (or any of the provisions in it, including the Services or the Purchase Orders), whether in order to include an additional service, function or responsibility to be performed by the Supplier for Simplyhealth under this Agreement, to amend the Services or the service levels attached to the Services or otherwise in a Purchase Order, Simplyhealth may at any time request, and the Supplier may at any time recommend, such Change and a Change Request shall be submitted by the party requesting/recommending (as applicable) the Change to the other. Such Change shall be agreed by the parties only once the Change Request is signed by both parties.\n[bookmark: _731f9a46-43d1-4d96-a80d-591e5b0e0692]Each Change Request shall conform to the requirements of Schedule 4.\n[bookmark: _6d2c5599-9998-4e29-a961-daea6c2a6460]Until such Change is made in accordance with clause 7.1, Simplyhealth and the Supplier shall, unless otherwise agreed in writing, continue to perform this Agreement in compliance with its terms prior to such Change.\n[bookmark: _b7787c42-953c-48de-9fbb-f65283a98688]Any discussions which may take place between Simplyhealth and the Supplier in connection with a request or recommendation before the authorisation of a resultant Change shall be without prejudice to the rights of either party.\n[bookmark: _1d016ce7-1449-4836-8870-8619a76125aa]Any Services or other work performed by the Supplier and/or the Supplier Personnel to Simplyhealth which have not been agreed in accordance with the provisions of this clause 7 shall be undertaken entirely at the expense and liability of the Supplier.\n[bookmark: _Toc256000008][bookmark: _Toc26196070][bookmark: _92d98048-87ac-4b1e-87e6-fa300e0dbd3a]Warranty\n[bookmark: _cd9fb215-78d0-468c-923d-700d0a97b3ac]The Supplier represents and warrants that:\n[bookmark: _be596c37-e69e-4e0b-8c13-f7d5dc2ba17e]it has the right, power and authority to enter into this Agreement and grant to Simplyhealth the rights (if any) contemplated in this Agreement and to perform the Services;\n[bookmark: _2ef443d6-8664-4090-bad9-bd6999c507d9]it understands Simplyhealth’s business and needs;\n[bookmark: _5bcd7a64-4572-4bb2-80ef-6a1e8d86fc7a]the Services shall be performed in accordance with Best Industry Practice;\tComment by Martin Philpott: Usually Best Practice would be too woolly but they have defined this above as:“Best Industry Practice means in relation to any undertaking and any circumstances, the highest degree of professionalism, skill, diligence, prudence and foresight which would be expected from an internationally recognised and market leading company engaged in the same type of activity under the same or similar circumstances and which is best in class” This doesn’t sound unreasonable but will hold us to high standards.  Could we push back with “Reasonable Industry Practice”? \tComment by Sacha Tomey: Yes, I think we should push for Reasonable – and without the definition..  Trouble with the definition is, if the agree and just change it in both places we’re still on the hook for the same high standards.\n[bookmark: _0fcc07fc-13c1-4fd8-a1a9-72cc6d9a465b]the Services performed and the Deliverables supplied shall comply with all applicable Laws;\n[bookmark: _da734a7b-b238-47c6-8de5-38e593621666]the Services performed and Deliverables supplied shall conform in all material respects to the corresponding Purchase Order and Specification;\n[bookmark: _8a16e563-fc17-4cf3-a681-97e51a4de694]the Services performed and Deliverables supplied shall not infringe the Intellectual Property Rights of any third party; and\n[bookmark: _Hlk26440750][bookmark: _cd954cef-d9bb-4ee5-abc3-fd428edef25f]the Services performed and Deliverables supplied shall be fit for any purpose held out by the Supplier.\n[bookmark: _8d3cf809-7359-4083-b949-1ae550c9947b]Without limiting any other remedies to which it may be entitled, Simplyhealth may reject any Services or Deliverables that do not comply with clause 8.1 and the Supplier shall, at Simplyhealth’s option, promptly remedy, re-perform or refund the Price of any such Services or Deliverables provided that Simplyhealth serves a written notice on Supplier within the Warranty Period that some or all of the Services or Deliverables (as the case may be) do not comply with clause 8.1, identifying in sufficient detail the nature and extent of the defects.\n[bookmark: _e717d0d2-f0fe-48a8-882f-5aa1bc1dd221]The provisions of this Agreement and the corresponding Purchase Order shall apply to any Services and related Deliverables that are remedied, re-performed or redelivered pursuant to clause 8.2.\n[bookmark: _f3424616-7a84-449e-b8a7-e1f8970b2cc7]The Supplier shall not be liable for a breach of clause 8.1 to the extent that such breach arises by reason of:\n[bookmark: _e51902ca-4844-46ac-b0df-60351bf96454] Simplyhealth’s wilful damage or negligence;\n[bookmark: _226c35c8-d534-45d4-9e43-fbe23f881f33] the Supplier’s use of Simplyhealth Materials; or\n[bookmark: _77d3a6a9-6294-4bb2-aa89-92301e285979]an event of Force Majeure.\n[bookmark: _673b0205-d250-46d8-9d90-d3e7660aeb41]The provisions of this clause 8 are in addition to, and are not exclusive of, any other rights and remedies to which Simplyhealth may be entitled, and the warranties and conditions implied by the Sale of Goods Act 1979 or the Supply of Goods and Services Act 1982 are not excluded.\n[bookmark: _Toc256000009][bookmark: _Toc26196071][bookmark: _37e6818e-1c9c-4c0c-9f4a-63e209bd4c4a]Price\n[bookmark: _3062e9e9-ec83-4bb9-b765-f582a24a4017]The Prices payable by Simplyhealth in respect of each Purchase Order for Services are contained in the applicable Statement of Work and/or Purchase Order.\n[bookmark: _e9310f26-79c9-45ca-a615-fe5a25b413d8]The Prices are inclusive of VAT.\n[bookmark: _c0f13d39-1d6d-4938-b421-a05df0929b66][bookmark: _085f5329-7ae8-435f-8a04-89146321cafa][bookmark: _9d67bde6-7f96-413b-a930-877fd82f40f0]Where the Prices are calculable on a time and materials basis, the Supplier will keep time sheets showing the hours worked by each of the Supplier Personnel in respect of the provision of the corresponding Services and will if so requested produce them to Simplyhealth for accounting purposes.\n[bookmark: _6b126477-83e2-4f50-b5d1-1949a6ac853f]Simplyhealth will be responsible for any reasonable out-of-pocket expenses incurred by the Supplier and the Supplier Personnel in the performance of its obligations under this Agreement and under the Purchase Orders provided the expenses are in accordance with Simplyhealth’s Expenses Policy. For the avoidance of doubt the Supplier must give Simplyhealth advance notice of any expenses before they are incurred.\n[bookmark: _Toc256000010][bookmark: _Toc26196072][bookmark: _10827a2c-2f69-4261-be29-9e19215984fe]Payment\n[bookmark: _04bc4273-3423-4ad3-901f-48a6ffffe103]The Supplier shall issue its invoice in respect of a Purchase Order, on a monthly basis.\nThe parties have agreed the following discount structure for invoices: \n(a)\tif the invoiced value passes £300,000 then a £15,000 credit note will be issued to Simplyhealth to use for the current invoice; and\n(b)\tif the invoiced value passes £600,000 then a further £20,000 credit note will be issued to Simplyhealth to use for the current invoice. \n[bookmark: _7a73f105-911a-488d-8728-4208cb1893e8]Simplyhealth shall pay all undisputed invoices:\n[bookmark: _d45dd087-b54e-4663-b905-9ce710c2a4fa]in full in cleared funds within 30 days of receipt of each invoice; and\n[bookmark: _c45566ef-9449-4cf5-be77-149f2f4e6130]to the bank account nominated by the Supplier in the Purchase Order.\n[bookmark: _65012707-7cbc-41e0-9528-b351edb68be3]Simplyhealth shall pay any applicable VAT to the Supplier on receipt of a valid VAT invoice.\n[bookmark: _c3688b9e-9f4d-4201-a1ab-a913aa40b1b1]Time of payment is not of the essence. Where sums due are not paid in full by the due date:\n[bookmark: _aa152b84-fbb0-49a4-a193-7258c34d5cc7]the Supplier may charge interest on such sums at 2 percentage points a year above the base rate of the Bank of England from time to time in force; and\n[bookmark: _d8e9d589-563b-4263-8a19-4e48dfefa702]interest shall accrue on a daily basis, and apply from the due date for payment until actual payment in full, whether before or after judgment.\n[bookmark: _Toc256000011][bookmark: _Toc26196073][bookmark: _ad746069-d5e0-4ccd-a7e4-6de56760d74e]Data protection\n[bookmark: _73707fb8-a2e7-49b7-9dc6-12366a631b09]Each party agrees that, in the performance of their respective obligations under this Agreement, it shall comply with the provisions of Schedule 3.\nFor the avoidance of doubt, a breach of this clause 11 shall be considered a material breach for the purposes of clause 16.2. \n[bookmark: _Toc256000012][bookmark: _Toc26196074][bookmark: _de402940-ce4a-485c-8cf5-17cdbaa8874a]Insurance\tComment by Cassie Oakley: Adatis to provide their insurance amounts. \tComment by Nick Baladi: Ask Louis\n[bookmark: _1473c6d9-5ead-4a92-abfc-b889ae76f204]The Supplier shall put in place and maintain the following insurance with a reputable insurer for the duration of this Agreement and for one year after its termination or expiry:\npublic and products liability insurance for not less than £[insert amount] in respect of each claim; \n[bookmark: _ea0ffa47-231f-4463-9a38-d33611ada3f1]property damage insurance for not less than £[insert amount] in respect of each claim;\n[bookmark: _992438d6-9cfc-4c67-9d33-d49d5da0b836]professional indemnity insurance for not less than £[insert amount] in respect of each claim; and\n[bookmark: _4ca86b17-5825-4169-86ae-98ed46189cf1][bookmark: _aaff1381-ead0-468f-8e12-90a51c68ba35]employer's liability insurance for not less than £5 million in respect of each claim. \n[bookmark: _04f4ea51-9bda-43ab-999a-abb1cbe3efbc]The Supplier undertakes that it shall not do or omit to do anything which might invalidate or adversely affect the insurance that the Supplier is obliged to maintain under clause 12.1.\n[bookmark: _ec8789a6-3f6c-4be6-8e65-cedc3308bdc0]The Supplier shall notify Simplyhealth immediately if anything occurs which has invalidated, or is likely to invalidate, the insurance held by the Supplier.\n[bookmark: _Toc256000013][bookmark: _Toc26196075][bookmark: _6d83f2a1-51c8-44c6-ae8a-67733c9fc469]Intellectual property rights\n[bookmark: _75487f0a-310d-4a84-90e8-2bab489e68e1]In consideration of the Prices payable under this Agreement (the receipt and sufficiency of which Supplier hereby acknowledges) and the parties' mutual obligations under this Agreement the Supplier assigns to Simplyhealth absolutely with full title guarantee all the present and future Intellectual Property Rights in the Services and Deliverables and all other materials created by the Supplier pursuant to this Agreement for Simplyhealth’s internal business purpose only.\nWhere any Intellectual Property Rights of the Supplier are incorporated into the Services and/or Deliverables the Supplier grants Simplyhealth the right to use the Supplier’s Intellectual Property Rights:\n(a)\t in perpetuity on an irrevocable basis as part of the Services and/or Deliverables for Simplyhealth’s internal business purposes only; or \n(b) \tpursuant to a separate agreement on mutual agreed terms and for both parties to maintain the confidentiality of the terms.  \nExcept as expressly agreed above in clauses 13.1 and 13.2, no Intellectual Property Rights of either party are transferred or licensed as a result of this Agreement. \n[bookmark: _41e9cd05-c32d-40a6-ac86-2a9d60bed7b9]The Supplier shall have no ongoing right to use or license or otherwise encumber or exploit the Intellectual Property Rights in the Services and the Deliverables (or permit others to do so) following receipt by it of the Prices payable under this Agreement.\n[bookmark: _97297b68-7126-47b4-ae96-85bf20d4ddaf]Subject to the foregoing, each party shall be entitled to use in any way it deems fit any skills, techniques or know-how acquired or developed or used in connection with the Services or otherwise in connection with this Agreement provided always that such skills, techniques or know-how do not infringe the other party's Intellectual Property Rights now or in the future or disclose or breach the confidentiality of the other party's Confidential Information.\n[bookmark: _Toc256000014][bookmark: _Toc26196076][bookmark: _c0d17187-da3c-4759-910f-c07d7ebddcf1]Limitation of liability\n[bookmark: _46f0f08e-c40f-4704-beb7-8cc5f96ae48b]The extent of the liability of each party under or in connection with this Agreement (regardless of whether such liability arises in tort, contract or in any other way and whether or not caused by negligence or misrepresentation) shall be as set out in this clause 14.\n[bookmark: _ae90bea4-dfae-4be1-9245-7e505e32b7a8]Subject to clauses 14.5, 14.6 and 15 the total liability of the:\n[bookmark: _062918fd-1019-4c37-8e67-df689d392c11]Supplier, howsoever arising under or in connection with this Agreement, shall not exceed the sum of £1 million; and\n[bookmark: _1ad14a07-ff69-4990-9007-978e59773743]Simplyhealth, howsoever arising under or in connection with this Agreement, shall not exceed the sum of £1 million (provided that this limitation shall not apply to the Prices (and VAT and interest thereon) due and payable by Simplyhealth to the Supplier under this Agreement).\n[bookmark: _b51ad946-4d35-429f-9fcb-96157f5a3fe4]Subject to clause 14.6, each party shall not be liable for consequential, indirect or special losses.\n[bookmark: _ab5cb910-cf64-45c0-8c67-0e8a9cc96241]Subject to clause 14.6, each party shall not be liable for any of the following (whether direct or indirect):\n[bookmark: _b66595ee-631d-475c-bbe6-059fc8b82097]loss of profit;\n[bookmark: _18a20d33-ca1b-4dae-a89c-5f11a1c6aabc]loss of use;\n[bookmark: _e328adac-b213-4844-8307-a5554acfcd9d]loss of production;\n[bookmark: _335be22e-ee85-4056-a863-64beafc0dec8]loss of contract;\n[bookmark: _93b72e4a-908b-4b59-9af9-5e14f529d4e0]loss of opportunity;\n[bookmark: _301b68e1-2fae-4a4b-ac05-e6afaa975920]loss of savings, discount or rebate (whether actual or anticipated);\n[bookmark: _88312191-549b-4661-abb8-d208877eacef]harm to reputation or loss of goodwill.\n[bookmark: _f929bebc-7c6d-4f23-b1c2-47940f343a8f]The limitations of liability set out in clause 14.2 shall not apply in respect of liability under any indemnities given by the Supplier under this Agreement and any sums payable in connection with such indemnities shall not be taken into account when aggregating the Supplier's liability under clause 14.1. \n[bookmark: _daf8e591-4229-4d1e-a4ac-cab1865d8fa9]Notwithstanding any other provision of this Agreement, the liability of the parties shall not be limited in any way in respect of the following:\n[bookmark: _7e784f00-79b3-4049-9ec5-cc18142d2f32]death or personal injury caused by negligence;\n[bookmark: _f68dcbbc-6cef-4c65-919d-e296cecc842d]fraud or fraudulent misrepresentation;\n[bookmark: _c785e43b-53a2-4bcf-a505-489452db061f]any other losses which cannot be excluded or limited by applicable law;\n[bookmark: _9c2363e5-0bf3-4d50-8e28-498ce013f8ed]any losses caused by wilful misconduct.\n[bookmark: _Toc256000015][bookmark: _Toc26196077][bookmark: _31e3b543-66bc-4fe0-84b1-a5ff271e1ea3]Indemnity\n[bookmark: _a985b726-be4b-4cfe-8150-3218c6b7036b]The Supplier shall indemnify Simplyhealth for any losses, damages, liability, costs and expenses (whether direct or indirect) including any interest, penalties, and legal and other professional fees and expenses incurred by it as a result of any action, demand or claim:\n[bookmark: _2aa93b93-fdad-4576-b378-9cdf308c4a95]that the provision of the Services or Deliverables infringes the Intellectual Property Rights of any third party (an IPR Claim);\n[bookmark: _15ce5065-8038-4812-bfa5-1a049c9d1948]that Simplyhealth is in breach of any applicable Law as a result of any act or omission of the Supplier;\nany claim made against Simplyhealth by a third party arising out of, or in connection with, the Services, to the extent that such claim arises out of a breach of the Agreement, any, negligent performance, any wilful misconduct and/or failure in performance of the Agreement by the Supplier, its employees, agents or subcontractors;\nthe Supplier or a subcontractor breaching Clauses 11 (Data Protection) or 19 (Confidentiality); \n[bookmark: _4fc256c8-a11f-494f-ace4-9c4dc4574c45]made against Simplyhealth by a third party arising from any defect in the performance of the Services or the provision of the Deliverables caused by the Supplier’s breach of this Agreement,\neach being a Claim.\n[bookmark: _c623f9c7-fabb-4469-9c91-61a85cbad352]In the event that Simplyhealth receives notice of any Claim, it shall:\n[bookmark: _d7f78c76-29f8-4243-99e4-7499c6a8961f]notify the Supplier in writing as soon as reasonably practicable;\n[bookmark: _49cc13ea-bf1c-46ce-97b8-e03eb6a5a069]not make any admission of liability or agree any settlement or compromise of the Claim without the prior written consent of the Supplier (such consent not to be unreasonably withheld or delayed);\n[bookmark: _fc6407e8-53b2-475e-af14-23917d16a08c]let the Supplier at its request and own expense have the conduct of or settle all negotiations and litigation arising from the Claim at its sole discretion provided that if the Supplier fails to conduct the Claim in a timely or proper manner Simplyhealth may conduct the Claim at the expense of the Supplier;\n[bookmark: _74ac8550-156b-4953-bbbf-7d4e815bd877]take all reasonable steps to minimise the losses that may be incurred by it or by any third party as a result of the Claim; and\n[bookmark: _2cabf309-bc9c-4e07-a0d1-0ddbf4beba48]provide the Supplier with all reasonable assistance in relation to the Claim (at Simplyhealth’s expense) including the provision of prompt access to any relevant premises, officers, employees, contractors or agents of Simplyhealth.\n[bookmark: _e78f5243-7bd6-45aa-9779-802cd697b3cb]If any IPR Claim is made or is reasonably likely to be made, the Supplier may at its option:\n[bookmark: _317587a9-a57e-4cfb-a619-510f3b7f2d71]procure for Simplyhealth the right to continue receiving the relevant Services or using and possessing the relevant Deliverables; or\n[bookmark: _0c05ab04-7e01-4884-9e25-31984dc05921]re-perform the infringing part of the Services or modify or replace the infringing part of the Deliverables so as to avoid the infringement or alleged infringement, provided the Services or Deliverables remain in conformance to the Specification.\n[bookmark: _2a0d75c6-0022-4e7c-8ab3-d8e88870d3aa]The Supplier's obligations under clause 15.1 shall not apply to Deliverables modified or used by Simplyhealth other than in accordance with this Agreement or the Supplier’s reasonable written instructions.\n[bookmark: _Toc256000016][bookmark: _Toc26196078][bookmark: _06e6cf5d-350a-46cf-a31a-b1df76304d32]Termination\n[bookmark: _af33455a-fc36-4f94-a947-04971ed46ab0]This Agreement may be terminated by either party giving not less than thirty (30) days’ notice in writing to the other party.\n[bookmark: _5f3d2991-7cf7-468f-9b8d-50b2e22daf36]Either party may terminate this Agreement at any time by giving notice in writing to the other party if:\n[bookmark: _1c0f662f-3cfb-4904-b230-1ebf26475893]the other party commits a material breach of this Agreement and such breach is not remediable;\n[bookmark: _f4f931af-2b70-46eb-bb8c-eb89e8bc931f]the other party commits a material breach of this Agreement which is not remedied within thirty (30) days of receiving written notice of such breach;\n[bookmark: _e2e734d6-b86c-46c7-9310-b12de211157c]any consent, licence or authorisation held by the other party is revoked or modified such that the other party is no longer able to comply with its obligations under this Agreement or receive any benefit to which it is entitled.\n[bookmark: _54245e6e-6841-430b-8101-e46f69c9a2c2]Either party may terminate this Agreement at any time by giving notice in writing to the other party if that other party:\n[bookmark: _29b6bc21-ddb9-4597-9230-af868c8005e8]stops carrying on all or a significant part of its business, or indicates in any way that it intends to do so;\n[bookmark: _7e7be0e0-b173-4c87-8f54-4bcba23580d1]is unable to pay its debts either within the meaning of section 123 of the Insolvency Act 1986 or if the non-defaulting party reasonably believes that to be the case;\n[bookmark: _4b5e5264-46b1-43a3-9842-777a8928d3c7]becomes the subject of a company voluntary arrangement under the Insolvency Act 1986;\n[bookmark: _877ede72-0403-4fcb-9c93-f8e09d069708]has a receiver, manager, administrator or administrative receiver appointed over all or any part of its undertaking, assets or income;\n[bookmark: _48a8a4a2-876d-434c-ba25-bc07e40552ac]has a resolution passed for its winding up;\n[bookmark: _51d48f33-c9df-4165-9695-a070587b3681]has a petition presented to any court for its winding up or an application is made for an administration order, or any winding-up or administration order is made against it;\n[bookmark: _75814e11-dc50-455b-8bc0-e300df0cc168]is subject to any procedure for the taking control of its goods that is not withdrawn or discharged within seven days of that procedure being commenced;\n[bookmark: _57b85cd5-84c5-4321-a5f3-1f79e450ab2a]has a freezing order made against it;\n[bookmark: _7c543496-2039-4aa8-a133-f9a3a1ab0a7e]is subject to any events or circumstances analogous to those in clauses 16.3 (a) to 16.3 (i) in any jurisdiction;\n[bookmark: _e9caa584-20f8-4ed6-b44a-3b68ad389519]takes any steps in anticipation of, or has no realistic prospect of avoiding, any of the events or procedures described in clauses 16.3 (a) to 16.3 (i) including for the avoidance of doubt, but not limited to, giving notice for the convening of any meeting of creditors, issuing an application at court or filing any notice at court, receiving any demand for repayment of lending facilities, or passing any board resolution authorising any steps to be taken to enter into an insolvency process.\n[bookmark: _cb773d74-61de-4d82-812b-925a0ffb89ac]The right of a party to terminate the Agreement pursuant to clause 16.3 shall not apply to the extent that the relevant procedure is entered into for the purpose of amalgamation, reconstruction or merger (where applicable) where the amalgamated, reconstructed or merged party agrees to adhere to this Agreement. \n[bookmark: _d1fbd882-7088-4e69-a75d-ae787f0345d1]On termination of this Agreement for any reason:\n[bookmark: _6b16da94-8f0e-4d24-84df-a57f9dad6dc4]the Supplier shall immediately stop the performance of all Services unless expressly requested otherwise in relation to all or part of the Services by Simplyhealth in writing;\n[bookmark: _e8e39fc1-2fb5-4854-910c-90767773af58]the Supplier shall promptly invoice Simplyhealth for all Services performed but not yet invoiced and/or refund any sums paid in advance for Services not performed;\n[bookmark: _69a627f9-e2da-4393-a7a1-7c3007e4e228]without prejudice to any additional obligations under Schedule 3, the parties shall within five Business Days return any materials of the other party then in its possession or control which the parties no longer have the right to have in their possession or control;\nthe Supplier shall undertake all acts reasonably requested by Simplyhealth in order to ensure a smooth handover of the Services prior to and after termination or expiry of the Agreement; \n[bookmark: _2b4d0e7d-2aef-4031-9a72-03a0dfcfbf98]all accrued rights and liabilities of the parties (including any rights in relation to breaches of contract) shall not be affected; and\n[bookmark: _495b2458-d013-4650-8e56-01de98162f41]all rights granted to the Supplier under this Agreement or any Purchase Order shall immediately cease.\n[bookmark: _d085b9a2-e844-4ea3-972b-5dc6c5c867bc]The following clauses of this Agreement shall survive termination, howsoever caused:\nclause 1 (definitions and interpretation); \n[bookmark: _9f389aef-29ef-431b-ae7c-c3d74456cda7]clause 8 (warranty);\n[bookmark: _963fcaab-d04c-4cc1-ab05-f24bf744ec15]clause 11 (data protection);\n[bookmark: _398b6b3b-75e7-4373-b34e-036b4b5e9bf3]clause 12 (insurance);\n[bookmark: _d3fc7314-47da-4de8-a597-69acd93621a6]clause 14 (limitation of liability);\n[bookmark: _c388e075-75ba-41d1-8f36-67159a6830ea]clause 15 (indemnity);\n[bookmark: _d568af09-b7e0-44dd-8a99-929916f83b68]clause 16.5 (consequence of termination);\n[bookmark: _b0d75676-3653-4c5a-b8a0-43bc2e793270]clause 17 (personnel);\n[bookmark: _9dc9f198-4c33-4c4f-9362-38f214304524]clause 18 (non-solicitation);\n[bookmark: _aec96fe0-3c49-4f9b-be01-e69e57f99cf6]clause 19 (confidential information);\n[bookmark: _3e18b3dc-12f1-43d2-bcd3-fa97870cbc4f]clause 23 (audits and investigations);\n[bookmark: _e04a1332-e7e7-43d8-83fa-93686968e116]clause 24 (dispute resolution);\n[bookmark: _c5732e30-b88a-45b6-88dd-11aee4f0d25f]clause 27 (notices);\n[bookmark: _721ee31d-b3b1-4d1c-a525-9699c0eede5b]clause 41 (third party rights);\n[bookmark: _ae5c876d-b437-43b2-8ed6-1b94a1fe75fe]clauses 42 and 43 (governing law and jurisdiction); and\n[bookmark: _54e212e3-44ef-4927-b64a-be7ee19f53e3]Schedule 3 (data protection)\ntogether with any other provision of this Agreement which expressly or by implication is intended to survive termination.\n[bookmark: _Toc256000017][bookmark: _Toc26196079][bookmark: _06e04dc1-689b-4e9e-a8f0-35c66742a8e8]Personnel and TUPE\nThe parties acknowledge that they have no intention that any Supplier Personnel shall transfer to Simplyhealth or any of its Affiliates or to any service provider to Simplyhealth or its Affiliates by virtue of TUPE on the termination or expiry of this Agreement.\nIf any Supplier Personnel does transfer to Simplyhealth or any of its Affiliates or to any service provider to Simplyhealth or its Affiliates under TUPE as a result of the termination or expiry of this Agreement the Supplier shall indemnify Simplyhealth and its Affiliates and on Simplyhealth’s written instruction Simplyhealth’s replacement service provider against all costs, expenses and liabilities arising on and from the point of transfer onwards in respect of the person’s subsequent employment and/or dismissal from employment.\n[bookmark: _985e90d7-e3f5-4589-bee4-9b257f654ece][bookmark: _919d17c9-bbbb-43f1-9f81-359d1d0ce1f7][bookmark: _946b85d2-8a4d-42e6-aecb-623a9409f3e9][bookmark: _afdcf346-06c2-4acb-ba42-9fd7ca2de785]The Supplier shall ensure that all Supplier Personnel comply with:\n[bookmark: _3e620897-c601-466b-962e-fffa6de2f0d7]any protocols, codes of conduct or procedures agreed between the parties to be applicable which may include the Policies and Procedures and any occupational health and health and safety requirements, building access and physical security policies, employee conduct requirements and environmental policies which are notified to the Supplier by Simplyhealth from time to time in writing; and\n[bookmark: _dab83376-7217-47d9-8bd5-a3184731e301]the Supplier’s obligations under this Agreement.\n[bookmark: _95a977ef-d4d9-4068-bd0b-f0e4725c73bf]To the extent legally permissible, the Supplier shall not employ as Supplier Personnel individuals whose previous background would reflect adversely upon Simplyhealth (including those individuals convicted of serious criminal offences) and the Supplier shall carry out such checks as are legally permissible to guard against this.\n[bookmark: _4462183b-71ae-4b5f-97fc-36845d62b076]The Supplier shall use best endeavours to minimise the turnover rate of Supplier Personnel and shall use best endeavours to minimise all disruption occasioned by Supplier Personnel turnover.\tComment by Martin Philpott: Usually would not sign up to “Best Endeavours” but this relates to staff turnover (and not deliverables|) so it’s possibly OK?Would like to amend to be reasonable endeavours ideally\tComment by Sacha Tomey: Hmm.  Not sure I agree with you.Best endevaours to minise turnover might mean we have to pay them £1m per year to keep them..More of an issue for me is the best endeavours to minimise disruption..  that does relate to deliverables.We need to push back on this\tComment by Nick Baladi: Change to Reasonable endeavours\n[bookmark: _cd1bc628-05dd-467a-bf55-d737cf72282a]The Supplier shall comply with all Laws applicable to the employment or retention of the Supplier Personnel, including work permits, immigration, customs, foreign payment or similar requirements and shall indemnify Simplyhealth without limit against any losses, damages, liability, costs and expenses (including professional fees) arising out of the Supplier’s failure to comply with this clause 17.6.\n[bookmark: _7e64f80c-8075-4795-ad2d-cc1c15c29fee]Simplyhealth reserves the right to demand the replacement at any time, without notice, of any Supplier Personnel if, in the reasonable opinion of Simplyhealth, the performance or conduct of such Supplier Personnel is or has been unsatisfactory or if a regulatory body requires Simplyhealth to do so, in which case the Supplier shall promptly remove the relevant Supplier Personnel from the provision of Services and, where required, promptly provide replacement Supplier Personnel. Simplyhealth shall provide such details as it is reasonably able to provide as to the basis for any required removal of Supplier Personnel. Simplyhealth shall not be charged for any training or other costs incurred due to the change.\n[bookmark: _75fd0e30-e225-49ad-aaaf-c311ff3e62ee]All Supplier Personnel whom the Supplier proposes to carry out work or perform duties under this Agreement and who shall be required, whilst carrying out some or all of that work or performing some or all of those duties, to:\n[bookmark: _0493efea-4e48-4aa8-9a4b-eaef8dd48ef7]enter the Performance Location;\n[bookmark: _c896ba19-ca1c-479b-84fd-d81c857bf806]work with Simplyhealth’s personnel for extended periods; or\n[bookmark: _af74582f-13a0-4147-9b50-6fb978729fc1]hold a particular kind of security clearance where details of such additional requirements have been notified to the Supplier by Simplyhealth;\nmust be authorised by Simplyhealth to carry out that work or perform those duties, such authorisation by Simplyhealth to be in compliance with Simplyhealth’s security policies from time to time in force and notified to the Supplier in writing.\n[bookmark: _f5e0d064-6d2e-49b9-b859-1e8d75239329]Subject to any duties of confidence, Simplyhealth shall provide to the Supplier, in the form reasonably required by the Supplier, such information as the Supplier reasonably requests, from time to time, for the purpose of allowing the Supplier to comply with any security requirements for the purposes of clause 17.8.\n[bookmark: _Toc256000018][bookmark: _Toc26196080][bookmark: _97a2ec9e-9dc0-4fa4-9e1f-ea78660013ca]Non-solicitation\nNeither party shall during the Restricted Period, solicit or entice away or attempt to entice any Restricted Person from the other party. For the avoidance of doubt, this clause shall not apply in respect of the recruitment of any Restricted Person who has responded to a publicly advertised role. \n[bookmark: _4b0ff7d8-dda4-43ec-950e-e38e9a1ed9c9][bookmark: _d1f43794-c901-4797-bfcb-e448c6e575e7][bookmark: _421e7b72-387e-4bec-86f7-53223f2e5630][bookmark: _ff02f125-7cc4-42fb-b2ca-ae13ff57d54a][bookmark: _Toc256000019][bookmark: _Toc26196081][bookmark: _9b69a72f-31d8-415a-9431-06ca366dba77]Confidential information\n[bookmark: _1f8740c5-286b-4a94-a081-9da7a16352b1]Each party undertakes that it shall keep any information that is confidential in nature concerning the other party and its Affiliates including, any details of its business, affairs, customers, clients, suppliers, plans or strategy (Confidential Information) confidential and that it shall not use or disclose the other party's Confidential Information to any person, except as permitted by clause 19.2.\n[bookmark: _5b63874d-dd2f-4275-9587-fbf5b15f9b8b]Subject to clause 19.6, a party may:\n[bookmark: _3aeeed9a-7c0e-4b2f-822e-853fd8530df4]disclose any Confidential Information to any of its employees, officers, representatives or advisers (Representatives) who need to know the relevant Confidential Information for the purposes of the performance of any obligations under this Agreement, provided that such party must ensure that each of its Representative to whom Confidential Information is disclosed is aware of its confidential nature and agrees to comply with this clause 19 as if it were a party;\n[bookmark: _f7f8b257-198f-4374-a183-aadb38396041]disclose any Confidential Information as may be required by law, any court, any governmental, regulatory or supervisory authority (including any securities exchange) or any other authority of competent jurisdiction to be disclosed; and\n[bookmark: _175c34e1-5c1a-4f29-ab1f-52ffed888208]use Confidential Information only to perform any obligations under this Agreement.\n[bookmark: _18fff99e-1b0e-4d9d-9688-c43c76245237]Each party recognises that any breach or threatened breach of this clause 19 may cause irreparable harm for which damages may not be an adequate remedy. Accordingly, in addition to any other remedies and damages, the parties agree that the non-defaulting party may be entitled to the remedies of specific performance, injunction and other equitable relief without proof of special damages.\nFor the avoidance of doubt, a breach of this clause 19 shall be considered a material breach for the purposes of clause 16.2. \n[bookmark: _041de2f6-b6f0-40b6-a2bf-3c4555c253db]This clause 19 shall bind the parties during the Term and for a period of five (5) years following termination of this Agreement.\n[bookmark: _b874d72e-0a57-4775-9fb7-b4f7c502e3f5]To the extent any Confidential Information is Personal Data (as defined in Schedule 3) such Confidential Information may be disclosed or used only to the extent such disclosure or use does not conflict with any of Schedule 3.\n[bookmark: _Toc256000020][bookmark: _Toc26196082][bookmark: _021361fc-7e01-4c89-9215-26f2a7ddc548]Anti-bribery\n[bookmark: _ec29e3ff-0dd3-4f49-991e-83f923b61c9e]For the purposes of this clause 20 the expressions ‘adequate procedures’ and ‘associated with' shall be construed in accordance with the Bribery Act 2010 and guidance published under it.\n[bookmark: _66693cb6-a5b6-430a-a324-d2f8d514cce4]The Supplier shall ensure that it and each person referred to in clauses 20.2 (a) to 20.2 (c) (inclusive) does not, by any act or omission, place Simplyhealth in breach of any Bribery Laws. The Supplier shall comply with all applicable Bribery Laws in connection with the performance of the Services and this Agreement, ensure that it has in place adequate procedures to prevent any breach of this clause 20 and ensure that:\n[bookmark: _df99915f-e818-4fbb-9cde-6a8773832b04]all of the Supplier Personnel and all direct and indirect subcontractors, suppliers, agents and other intermediaries of the Supplier;\n[bookmark: _0610583b-a5df-4118-a3cc-b138b8f7f7d0]all others associated with the Supplier; and\n[bookmark: _35ffb70b-9355-41bc-96ee-40b3acefca7a]each person employed by or acting for or on behalf of any of those persons referred to in clauses 20.2 (a) and/or 20.2 (b),\ninvolved in performing services for or on behalf of the Supplier or with this Agreement so comply.\n[bookmark: _a9d1ef33-0c6a-4b14-b8c1-7d4d9728f20d]Without limitation to clause 20.2, the Supplier shall not make or receive any bribe (which term shall be construed in accordance with the Bribery Act 2010) or other improper payment or advantage, or allow any such to be made or received on its behalf, either in the United Kingdom or elsewhere, and will implement and maintain adequate procedures to ensure that such bribes or improper payments or advantages are not made or received directly or indirectly on its behalf.\n[bookmark: _4cac17a7-4ab5-4781-bf7b-a64688486d4a]The Supplier shall immediately notify Simplyhealth as soon as it becomes aware of a breach or possible breach of any of the requirements in this clause 20.\n[bookmark: _fa9400ec-b968-4dd9-83e8-2198d62de016]Any breach of this clause 20 by the Supplier shall be deemed a material breach of this Agreement that is not remediable and entitle Simplyhealth to immediately terminate this Agreement by notice under clause 16.2 (a).\n[bookmark: _Toc256000021][bookmark: _Toc26196083][bookmark: _2309fe96-c5cd-4ad6-bf5f-f58f7d089b51]Modern slavery\n[bookmark: _b236a618-d466-4b9c-aca0-03fc58246458]The Supplier undertakes, warrants and represents that:\n[bookmark: _693d0763-7f04-4997-9e6e-c4dd79b38c09]neither the Supplier nor any of its officers, employees, agents or subcontractors has:\n[bookmark: _f93ca551-37c4-4f3c-b482-125f27a45ffc]committed an offence under the Modern Slavery Act 2015 (an MSA Offence); or\n[bookmark: _924bc9b6-817b-4afb-9123-ae06091a150a]been notified that it is subject to an investigation relating to an alleged MSA Offence or prosecution under the Modern Slavery Act 2015; or\n[bookmark: _0d4115c6-0f40-45ce-a1eb-580c72bec0e1]is aware of any circumstances within its supply chain that could give rise to an investigation relating to an alleged MSA Offence or prosecution under the Modern Slavery Act 2015;\n[bookmark: _18ebe584-8a49-4f0d-a7d3-14b6c70cb87d]it shall comply with the Modern Slavery Act 2015 and the Modern Slavery Policy; and\n[bookmark: _0fbaeaac-9b82-46df-b276-12ab82b542a6]it shall notify Simplyhealth immediately in writing if it becomes aware or has reason to believe that it, or any of its officers, employees, agents or subcontractors have breached or potentially breached any of Supplier’s obligations under Clause 21. Such notice to set out full details of the circumstances concerning the breach or potential breach of Supplier’s obligations.\n[bookmark: _93d43463-3e43-478f-8925-1d343b58f7f0]Any breach of clause 21.1 by the Supplier shall be deemed a material breach of the agreement and shall entitle Simplyhealth to immediately terminate this Agreement by notice under clause 16.2 (a).\n[bookmark: _Toc256000022][bookmark: _Toc26196084][bookmark: _788d8053-6cf2-481b-9003-1321d4f67877]Anti-tax evasion facilitation\nThe Supplier shall not engage in any activity, practice or conduct which would constitute either:\na UK tax evasion facilitation offence under section 45(1) of the Criminal Finances Act 2017; or\na foreign tax evasion facilitation offence under section 46(1) of the Criminal Finances Act 2017.\nThe Supplier shall:\nhave and maintain in place throughout the term of this Agreement such policies and prevention procedures as are both reasonable to prevent the facilitation of tax evasion by another person (including without limitation any Supplier Personnel or agent of the Supplier) and to ensure compliance with clause 22.1;\npromptly report to Simplyhealth any request or demand from a third party to facilitate the evasion of tax within the meaning of Part 3 of the Criminal Finances Act 2017, in connection with the performance of this Agreement; and\nensure that any person associated with the Supplier who is performing services and providing goods in connection with this Agreement does so only on the basis of a written contract which imposes on and secures from such person terms equivalent to those imposed on the Supplier in clauses 22.1 and 22.2. The Supplier shall be responsible for the observance and performance by such persons of such terms, and shall be directly liable to Simplyhealth for any breach.\nFor the purposes of clause 22.2, the meaning of reasonable prevention procedures shall be determined in accordance with any guidance issued under section 47 of the Criminal Finances Act 2017 and a person associated with the Supplier includes any subcontractor of the Supplier.\n[bookmark: _9c15b38a-0c27-44e1-824f-193bbb04d56f][bookmark: _26a0b875-64b8-40c5-b174-908ed5b17138][bookmark: _8e2080d8-1966-4345-9d7a-57aca74f8fc4][bookmark: _04a9bd1a-662f-401b-99f8-3b4b16471d6d][bookmark: _37df8f63-6030-4409-8afc-ed8ec30b3e5e][bookmark: _dd825be0-b2fe-42c3-8567-08865ccba29a][bookmark: _715c6512-914a-40ae-8ec5-94a9fbf5d43f][bookmark: _b4f9b69e-121b-4645-9395-4d3627107169][bookmark: _8c84385b-5f52-4d7f-abd3-9f6e4f390dc4][bookmark: _a92436f2-497a-4273-8402-8e24282468ea][bookmark: _42e1b55f-27bf-4e86-8b78-84d945e7e325][bookmark: _679742cc-559c-423b-92dd-1f21ea55bbff][bookmark: _22487269-5956-4182-90ed-a8fe9c19ad63][bookmark: _0ae42895-4c4d-4ad1-9f38-124058d1d561][bookmark: _957ee9c2-ccb5-4424-a2ee-8c9588760d90][bookmark: _1ac0b22f-0f80-4a94-a36e-b91c9e89be42][bookmark: _fac90781-8678-4c72-9174-00855182fc28][bookmark: _a496abff-7841-430b-a777-01311fcdc6d9][bookmark: _69b5f8e4-97f4-45a6-a768-f0171882cd6e][bookmark: _cd8eb9c4-2a6f-4967-9188-c5d01f3702e5][bookmark: _d5f45259-d427-400e-97c3-98ff440dd7fb][bookmark: _ff7b8a61-d7de-46da-bfc3-ef7daa606ec8][bookmark: _600bdddf-53f0-4214-87c2-ae106e15a9df][bookmark: _397ccb54-f606-463e-9682-eed7ae8faa87][bookmark: _47c8ad74-246a-4b00-b2d9-58d3c29432f6][bookmark: _2ee65dc0-d2fb-4466-8dd4-f0ec065e88e2][bookmark: _8f984a20-5480-4b60-8b2c-ee147176cffd][bookmark: _ab38c7d0-bc29-4286-a066-70fda480886e][bookmark: _a53a75f6-43f5-4227-82c0-409ed352eb1a][bookmark: _c9fcfb01-6f70-4ed5-a574-a70d7309a1f0][bookmark: _94b8229c-e92e-4f47-9e96-6a4b96e98499][bookmark: _ec3840da-3f94-419b-82ab-d33cfe54c7e8][bookmark: _45b9fe37-c3f1-46a4-a63d-226913a33b8c][bookmark: _d7a188cd-66c2-4b56-8516-1394e5208cac][bookmark: _f17b9d96-acf7-4005-886e-634960dd5106][bookmark: _Toc256000023][bookmark: _Toc26196085][bookmark: _ddbf01a0-bfa9-4f42-b394-059b0909bd68]Audits and investigations\n[bookmark: _d37045a6-57ba-4431-8e1a-a7fb54194b71]The Supplier shall allow Simplyhealth and/or its agents to access, inspect and audit the Supplier’s records, accounts and other relevant information and premises (including allowing copying of documents):\n[bookmark: _153adf80-2d53-48ac-aa04-2f9a5279d7e5]during normal business hours on Business Days and subject to a minimum of seven (7) Business Days’ notice; and\n[bookmark: _1ea6fb40-1bc1-4bc6-b3df-4de0250d6e9d]not more often than two times in any rolling 12-month period;\nto the extent this is reasonably required for the purpose of verifying the Supplier’s compliance with its obligations under this Agreement. Where such access, inspection or audit is required by an official government regulator, the Supplier shall allow such inspection or audit at any time and there shall be not be a limit to the number of such inspections or audits that can be undertaken.\n[bookmark: _806b38fc-329e-4edf-bc30-327085f4771c][bookmark: _e77c5d18-bd5e-419f-80ee-f8a59934988d][bookmark: _24e09889-b5d1-4df2-9c08-a7cbf1d936ad][bookmark: _15ba1d93-7868-48b1-9263-5e88a7ea88fb][bookmark: _2711d9fb-b32a-472d-9e03-72c5d6622051]The audit rights under this clause 23 are in addition, and without prejudice, to the further audit or inspection obligations of the Supplier or rights of Simplyhealth under Schedule 3 and each may be exercised separately.\n[bookmark: _Toc256000024][bookmark: _Toc26196086][bookmark: _28232f84-830c-47c7-a218-a11d342017f0]Dispute resolution\n[bookmark: _Ref4541868]Without prejudice to the express rights of termination set out in this Agreement, both parties will attempt in good faith to resolve any disputes promptly by negotiations between those representatives of the parties who have authority to settle the dispute. This clause 24.1 is without prejudice to either party’s right to seek interim relief against the other party (such as an injunction) through the English Courts to protect its rights and interests, or to enforce the obligations of the other party.\nExcept in relation to injunctive relief under clause 24.1, neither party may commence any court proceedings in relation to any dispute until it has attempted to settle the dispute in accordance with this clause 24.\n[bookmark: _2f3c8aa6-0520-46b0-a1f1-feca5943526a][bookmark: _57fd6441-e245-4cca-9939-9f392e37a755][bookmark: _e250ded5-48d3-4983-9d62-72a51ae9c04b][bookmark: _f58f0af9-3838-44fd-baca-c6c0a0c35353][bookmark: _7a3fbca7-345a-4c7d-b441-1b8dc339669c][bookmark: _287795c4-d50b-41bc-b404-5a24d19727d0][bookmark: _bb1a2739-5f8c-43b2-8caf-e78898dbee42][bookmark: _5791d057-e1b2-4bc7-9acd-1fef2c3ba0ea][bookmark: _Toc256000025][bookmark: _Toc26196087][bookmark: _8d72babd-387e-44c9-9ef2-2d73c3540200]Force majeure\n[bookmark: _955e9534-95bf-4521-a0d1-8737213146f7]In this clause 'Force Majeure' means an event or sequence of events beyond a party's reasonable control preventing or delaying it from performing its obligations under this Agreement. \n[bookmark: _87eee950-980d-47ab-92d9-f7bbd249ddef]A party shall not be liable if delayed in or prevented from performing its obligations under this Agreement due to Force Majeure, provided that it:\n[bookmark: _04b6dc13-9270-4225-aae0-1c84e9c0a464]promptly notifies the other of the Force Majeure event and its expected duration; and\n[bookmark: _e4799f3a-b93f-4005-a637-22584148a7fd]uses reasonable endeavours to minimise the effects of that event.\n[bookmark: _ad33273b-706e-41e6-b6a7-cb7d49c32d85]If, due to Force Majeure, a party:\n[bookmark: _6f06b225-6cb3-47ed-9ac0-38a63fadc90d]is unable to perform a material obligation; or\n[bookmark: _649a1a6f-1732-4488-9ce4-241519f78306]is delayed in or prevented from performing its obligations for a continuous period of more than one (1) month,\nthe other party may terminate this Agreement on not less than four (4) weeks’ written notice.\n[bookmark: _Toc256000026][bookmark: _Toc26196088][bookmark: _ace53f26-2f8f-4a6c-bfaf-5c93c35c75d3]Entire agreement\n[bookmark: _f92abec0-32e4-406b-b4ca-27c303eb95d2]The parties agree that this Agreement and the Purchase Orders entered into pursuant to it constitutes the entire agreement between them and supersedes all previous agreements, understandings and arrangements between them, whether in writing or oral in respect of its subject matter.\n[bookmark: _560fa89f-0d1b-4d57-a397-4aa28a42d5a4]Each party acknowledges that it has not entered into this Agreement and the Purchase Orders entered into pursuant to it in reliance on, and shall have no remedies in respect of, any representation or warranty that is not expressly set out in this Agreement and the Purchase Orders entered into pursuant to it. No party shall have any claim for innocent or negligent misrepresentation on the basis of any statement in this Agreement.\n[bookmark: _ec22fd5b-198d-4f67-9b5c-5b6ca48fc9c0]Nothing in this Agreement purports to limit or exclude any liability for fraud.\n[bookmark: _Toc256000027][bookmark: _Toc26196089][bookmark: _60e7320e-0e97-4408-ae39-eae771680ee8]Notices\n[bookmark: _f2085419-e8a5-4abe-9e17-7c96b28574e6]Any notice given by a party under this Agreement shall:\n[bookmark: _16d41682-80d4-4ef5-85bd-bfad58948a3f]be in writing and in English;\n[bookmark: _fd57faac-5bac-4e4e-893b-6584ccdf3967]be signed by, or on behalf of, the party giving it (except for notices sent by email); and\n[bookmark: _3b4f813c-96b8-4c32-afd0-46c3315dbce9]be sent to the relevant party at the address set out in clause 27.3.\n[bookmark: _1bbb5add-601f-4008-82b4-d0727f11862f]Notices may be given, and are deemed received:\n[bookmark: _5c671116-53d6-46da-9c48-518e95f2f27c]by hand: on receipt of a signature at the time of delivery;\n[bookmark: _8b8e3727-71a3-4cf5-a30d-2e1a1fa21c54]by Royal Mail Recorded Signed For post: at 9.00 am on the second Business Day after posting; and\n[bookmark: _181c7d5e-2429-4473-916a-70d06314a308]by email (provided confirmation is sent by first class post): on receipt of a delivery email from the correct address.\n[bookmark: _3443b087-7fb8-41b5-a7d4-b999eb337030]Notices shall be sent to:\n[bookmark: _86b14e59-eb05-41d8-a85e-2775c7af49c4]Adatis for the attention of Martin Philpott at:\nBroadmede House, Farnham Business Park, Weydon Lane, Farnham GU9 8QT; \n[bookmark: _GoBack]Mand\n[copied to [insert name] at [insert address];]\n[bookmark: _b2c999c5-9dd7-469b-b2a5-f2452046128c]Simplyhealth Group Limited, addressed for attention of the Legal Team at:\nHambleden House, Waterloo Court, Andover, SP10 1LQ; \nLegalWeb@simplyhealth.co.uk; and\ncopied to Procurement Team at Group.procurement@simplyhealth.co.uk. \n[bookmark: _2467ba99-f840-4227-9771-4d0c6ab69e23]Any change to the contact details of a party as set out in clause 27.3 shall be notified to the other party in accordance with clause 27.3 and shall be effective:\n[bookmark: _6c37eda1-95c2-4a6d-81b0-aec66cf228a0]on the date specified in the notice as being the date of such change; or\n[bookmark: _366aa573-087f-4f89-8d0a-64a6abc38965]if no date is so specified, five (5) Business Days after the notice is deemed to be received.\n[bookmark: _23ce0b7e-6c8a-4114-901e-ce9be339329a]All references to time are to the local time at the place of deemed receipt.\n[bookmark: _d08a3fb2-14ea-44c9-81c0-297bafc1be95]This clause does not apply to notices given in legal proceedings or arbitration.\n[bookmark: _8122498c-008a-475e-aa96-8a294b5fef1e]A notice given under this Agreement is not validly served if only sent by email.\n[bookmark: _Toc256000028][bookmark: _Toc26196090][bookmark: _b46fd1b7-7b8f-4495-8744-ce519d72787b]Announcements\n[bookmark: _aa7f772d-7814-44f7-9cad-e8569f3cd44c]Subject to clause 28.2, no announcement or other public disclosure concerning this Agreement or any of the matters contained in it shall be made by, or on behalf of, a party without the prior written consent of the other party (such consent not to be unreasonably withheld or delayed). \n[bookmark: _fd17a385-bb70-45d2-a5e5-8638b213d02d]If a party is required to make an announcement or other public disclosure concerning this Agreement or any of the matters contained in it by law, any court, any governmental, regulatory or supervisory authority (including any recognised investment exchange) or any other authority of competent jurisdiction, it may do so. Such a party shall:\n[bookmark: _fea936b0-3c07-4765-a211-9b2edf30901e]notify the other party as soon as is reasonably practicable upon becoming aware of such requirement to the extent it is permitted to do so by law, by the court or by the authority requiring the relevant announcement or public disclosure;\n[bookmark: _7a31c9cb-4744-43c3-9dbf-2b62d9eb6e18]make the relevant announcement or public disclosure after consultation with the other party so far as is reasonably practicable; and\n[bookmark: _71b9ac70-1a53-4fd7-8891-455e5528f28c]make the relevant announcement or public disclosure after taking into account all reasonable requirements of the other party as to its form and content and the manner of its release, so far as is reasonably practicable.\n[bookmark: _Toc256000029][bookmark: _Toc26196091][bookmark: _8b77f138-dd02-43d5-ac94-7218b66394e2]Further assurance\nThe Supplier shall at the request of Simplyhealth, and at the cost of the Supplier, do all acts and execute all documents which are necessary to give full effect to this Agreement.\n[bookmark: _Toc256000030][bookmark: _Toc26196092][bookmark: _b3a57bb1-d99b-4c10-a298-52705cdc24f1]Variation\nNo variation of this Agreement shall be valid or effective unless it is in writing, refers to this Agreement and is duly signed or executed by, or on behalf of, each party.\n[bookmark: _Toc256000031][bookmark: _Toc26196093][bookmark: _ac0078ef-5b63-4ffc-a253-0ba7db6092dd]Assignment\n[bookmark: _7d3ca770-3261-4239-95ae-3a9982810656]The Supplier may not assign, subcontract or encumber any right or obligation under this Agreement, in whole or in part, without Simplyhealth’s prior written consent.\n[bookmark: _Toc256000032][bookmark: _Toc26196094][bookmark: _3be3ca5f-b89d-41d7-9f62-8d1ad0cef091]Set off\nEach party shall pay all sums that it owes to the other party under this Agreement without any set-off, counterclaim, deduction or withholding of any kind, save as may be required by Law.\n[bookmark: _Toc256000033][bookmark: _Toc26196095][bookmark: _74623044-1f1b-4176-aae1-ad155d4871d6]No partnership or agency\nThe parties are independent businesses and are not partners, principal and agent or employer and employee and this Agreement does not establish any joint venture, trust, fiduciary or other relationship between them, other than the contractual relationship expressly provided for in it. None of the parties shall have, nor shall represent that they have, any authority to make any commitments on the other party's behalf.\n[bookmark: _Toc256000034][bookmark: _Toc26196096][bookmark: _005d7b30-f78f-4bae-a222-7a89f0ed163e]Equitable relief\nEach party recognises that any breach or threatened breach of this Agreement may cause the other party irreparable harm for which damages may not be an adequate remedy. Accordingly, in addition to any other remedies and damages available to the other party, each party acknowledges and agrees that the other party is entitled to the remedies of specific performance, injunction and other equitable relief without proof of special damages.\n[bookmark: _Toc256000035][bookmark: _Toc26196097][bookmark: _c55bfb77-776e-45b7-baa6-8888abc59117]Severance\n[bookmark: _d122e190-79b0-43fa-add3-77574d3c4fb3]If any provision of this Agreement (or part of any provision) is or becomes illegal, invalid or unenforceable, the legality, validity and enforceability of any other provision of this Agreement shall not be affected.\n[bookmark: _a791ef3e-c959-4452-b818-1b3cbf29310d]If any provision of this Agreement (or part of any provision) is or becomes illegal, invalid or unenforceable but would be legal, valid and enforceable if some part of it was deleted or modified, the provision or part-provision in question shall apply with such deletions or modifications as may be necessary to make the provision legal, valid and enforceable. In the event of such deletion or modification, the parties shall negotiate in good faith in order to agree the terms of a mutually acceptable alternative provision.\n[bookmark: _Toc256000036][bookmark: _Toc26196098][bookmark: _504f1173-e3d1-4c3a-8d3c-179cee0aed0a]Waiver\n[bookmark: _b7e2b7f9-814d-485c-83fb-f3b078d87b73]No failure, delay or omission by either party in exercising any right, power or remedy provided by law or under this Agreement shall operate as a waiver of that right, power or remedy, nor shall it preclude or restrict any future exercise of that or any other right, power or remedy.\n[bookmark: _c1fd6cf3-1e36-4177-a60a-19efa2a62d65]No single or partial exercise of any right, power or remedy provided by law or under this Agreement shall prevent any future exercise of it or the exercise of any other right, power or remedy.\n[bookmark: _247ae10c-23b7-4857-9629-e8bbca2ff291]A waiver of any term, provision, condition or breach of this Agreement shall only be effective if given in writing and signed by the waiving party, and then only in the instance and for the purpose for which it is given.\n[bookmark: _Toc256000037][bookmark: _Toc26196099][bookmark: _9574ff5d-6413-4ab8-bac9-9327a405c0a9]Compliance with law\nEach party shall comply and shall (at its own expense unless expressly agreed otherwise) ensure that in the performance of its duties under this Agreement, its employees, agents and representatives will comply with all applicable laws and regulations, provided that neither party shall be liable for any breach of this clause 37 to the extent that such breach is directly caused or contributed to by any breach of this Agreement by the other party (or its employees, agents and representatives).\n[bookmark: _Toc256000038][bookmark: _Toc26196100][bookmark: _3b79ca87-d995-4bf9-88eb-8e54f3f8ec81]Conflicts within agreement\n[bookmark: _f673a982-f5f5-4d71-a685-cf61bbcc6166]In the event of any conflict or inconsistency between different parts of this Agreement, the following descending order of priority applies:\n[bookmark: _e1a3a66f-c6bb-49e3-98c5-5567bd85eb04]the terms and conditions in the main body of this Agreement and Schedule 3;\n[bookmark: _5bb9cb84-6632-4cd9-a76e-a98e00f04660]the other Schedules; and \n[bookmark: _2072dd55-0c78-45e5-94d2-c64004b5cde0]the Purchase Order.\n[bookmark: _922df114-4e92-4a3d-a9c8-441f8f235614]Subject to the above order of priority between documents, later versions of documents shall prevail over earlier ones if there is any conflict or inconsistency between them.\n[bookmark: _Toc256000039][bookmark: _Toc26196101][bookmark: _89c62c04-589a-45b6-801f-40e50647cfcc]Counterparts\n[bookmark: _a1dc3b41-6926-4124-a177-538aada4b676]This Agreement may be signed in any number of separate counterparts, each of which when signed and dated shall be an original, and such counterparts taken together shall constitute one and the same agreement.\n[bookmark: _82ed86c3-0af2-496d-8dd8-c647cd2d8ce5]Each party may evidence their signature of this Agreement by transmitting by email a signed signature page of this Agreement in PDF format together with the final version of this Agreement in PDF or Word format, which shall constitute an original signed counterpart of this Agreement. Each party adopting this method of signing shall, following circulation by email, provide the original, hard copy signed signature page to the other parties as soon as reasonably practicable.\n[bookmark: _Toc256000040][bookmark: _Toc26196102][bookmark: _b1d8786a-b40b-4481-9770-84f3f59e3e3e]Costs and expenses\nEach party shall pay its own costs and expenses incurred in connection with the negotiation, preparation, signature and performance of this Agreement (and any documents referred to in it).\n[bookmark: _Toc256000041][bookmark: _Toc26196103][bookmark: _1f928bc6-1782-477a-a982-861b5222e452]Third party rights\n[bookmark: _88303802-a7e4-49dc-ab63-b456661ffe2d]Except as expressly provided for in clause 41.2, a person who is not a party to this Agreement shall not have any rights under the Contracts (Rights of Third Parties) Act 1999 to enforce any of the provisions of this Agreement.\n[bookmark: _f1f5e1b9-3759-46ea-9a9d-5ce25316cf68]The Affiliates of Simplyhealth shall have the right to enforce the provisions of this Agreement.\n[bookmark: _Toc256000042][bookmark: _Toc26196104][bookmark: _43b945a6-d0ae-4151-925f-9228698bbc7d]Governing law\nThis Agreement and any dispute or claim arising out of, or in connection with, it, its subject matter or formation (including non-contractual disputes or claims) shall be governed by, and construed in accordance with, the laws of England and Wales.\n[bookmark: _Toc256000043][bookmark: _Toc26196105][bookmark: _bcc636ff-acc4-4b29-bd6b-260576531983]Jurisdiction\nThe parties irrevocably agree that the courts of England and Wales shall have exclusive jurisdiction to settle any dispute or claim arising out of, or in connection with, this Agreement, its subject matter or formation (including non-contractual disputes or claims).\nAgreed by the parties on the date set out at the head of this Agreement.\n\tSigned by Nick Baladi                           \n\t.................................\n\n\tfor and on behalf of\n\tDirector OR Authorised signatory\n\n\tAdatis Group Limited\n\t\n\n\nand\n\tSigned by Debbie Beavan\n\t.................................\n\n\tfor and on behalf of\n\tAuthorised signatory\n\n\tSimplyhealth Group Limited\n\t\n\n\n[bookmark: _8e16d71f-12fc-4147-a4e5-e175f6a55016][bookmark: _Toc256000044][bookmark: _Toc256000106][bookmark: _Toc11940738]\n[bookmark: _Toc256000045][bookmark: _Toc26196107]Services\tComment by Martin Philpott: This should be contained within the individual SOW’s and not here IMODoes this match what we will be doing, I thought we are doing a Discovery Phase? it looks very vague and comprehensive.I think it should be removed and a reference added to the SOW’s.\tComment by Sacha Tomey: Agree.  I don’t like this in here either\tComment by Nick Baladi: We propose removing this section is removed as the services detailed in the SOW\nConsultancy & business analysis includes but shall not be limited to:\n· The organisation and people aspects to support BI and MI \n· The processes and roles to support data ownership \n· The processes, roles, and responsibilities to support data quality management \n· The operating model to govern the supply and demand of MI and BI within the organisation \nThe communication and learning and development requirements to support business change initiatives relating to BI and MI \n· The data management policy \n· The initiation and creation of a business glossary and data dictionary \n· The initiation and creation of a reporting catalogue and concurrent report rationalisation \n\nBusiness Intelligence consultancy includes but shall not be limited to:\n· Infrastructure setup and configuration\n· Security set up\n· Framework deployment and setup\n· Enterprise Data Warehouse build\n· Ingestion of data from source systems\n· Functional testing\n· UAT support\n· Knowledge transfer and handover\n· Architecture development\n[bookmark: _1ee4e1ad-59fc-483e-9c0e-0002100d3bec][bookmark: _Toc256000046][bookmark: _Toc256000108][bookmark: _Toc11940740][bookmark: _Toc26196108]\n[bookmark: _8d51571c-269d-4cb1-9e03-c991ac1c2516][bookmark: _Toc256000048][bookmark: _Toc256000110][bookmark: _Toc11940742][bookmark: _13d6cdcf-3d18-4871-a66a-a03d786a9394][bookmark: _Toc256000050][bookmark: _Toc256000112][bookmark: _Toc11940744][bookmark: _17a2a7fd-2e9a-43a1-891e-4da31a4151b7][bookmark: _Toc256000052][bookmark: _Toc256000114][bookmark: _Toc11940746][bookmark: _Toc26196109]Simplyhealth IT Access Policy\n1. [bookmark: _Toc26196110]The purpose of the Simplyhealth IT Access Policy (the policy) is to establish Supplier responsibilities and the rules for Supplier access to Simplyhealth's information systems. Supplier access to Simplyhealth's information systems is granted solely for the work carried out under the Contract and for no other purposes.\n\n2. [bookmark: _Toc26196111]Simplyhealth IT will provide a technical point of contact for the Supplier.  The  point  of contact  will work with  the Supplier to make  certain  the Supplier is  fully  aware of,  and  in  compliance  with,  this policy.\n\n3. [bookmark: _Toc26196112]The Supplier's personnel, agents and/or subcontractors with access to Simplyhealth's information systems must be made aware of his/her responsibilities pursuant to this policy and the Contract between the Supplier and Simplyhealth.\n\n4. [bookmark: _Toc26196113]If appropriate, regular work hours and duties will be defined in the Contract.  Any  work performed and   any access   made   to  Simplyhealth's  information  systems  outside  of  defined parameters, and/or performance of the Services (including,  without  limit,  with  regard  to  hours  and   duties)  must  be  approved  in  writing  by appropriate Simplyhealth  IT management via the technical point of contact.\n\n5. [bookmark: _Toc26196114]Access  to  Simplyhealth  Information  systems  will  only  be  allowed  at  the  date  / time agreed between the Supplier and the technical point of contact.\n\n6. [bookmark: _Toc26196115]Simplyhealth will provide the vendor with uniquely identifiable access. The Supplier is responsible for maintaining the security of its password(s), which must be complex in accordance with industry best practice standards.\n\n7. [bookmark: _Toc26196116]All Supplier equipment  and  accounts  on  the Simplyhealth  network  will  remain disabled,  except when  in  use for authorised maintenance.\n\n8. [bookmark: _Toc26196117]Supplier personnel, agents and/or subcontractors must not seek to access Simplyhealth's information systems other than those for which they are to provide support to under the Contract.\n\n9. [bookmark: _Toc26196118]Supplier personnel, agents and/or subcontractors must adhere to this policy at all times. \n\n10. [bookmark: _Toc26196119]Supplier personnel, agents and/or subcontractors must  report  all  security  incidents  directly  to either  their technical  point  of contact or the IT Service  Desk (01264 342422).\n\n11. [bookmark: _Toc26196120]Upon termination of the Contract or at the request of Simplyhealth, the Supplier must surrender all Simplyhealth badges, access cards, equipment and supplies immediately. Equipment and/or supplies to be retained by the vendor must be documented and authorised by Simplyhealth IT management.\n\n12. [bookmark: _Toc26196121]All software used by the Supplier in providing Services to Simplyhealth must be properly inventorised and licensed.\n\n13. [bookmark: _Toc26196122]Any Supplier computer/laptop/PDA/tablet PC that is connected to Simplyhealth systems must have up-to-date virus protection and patches.\n\n14. [bookmark: _Toc26196123]Simplyhealth shall monitor the Supplier’s access and/or activities within any production and development environments. \n\n[bookmark: _7be0b444-0373-401e-b3b3-8cf3a3a54b29][bookmark: _Toc256000054][bookmark: _Toc256000116][bookmark: _Toc11940748][bookmark: _Toc26196124]\n[bookmark: _Toc256000055][bookmark: _Toc26196125]Data protection\n[bookmark: _Toc256000056][bookmark: _Toc26196126][bookmark: _4cc193f3-3694-4926-a548-45bd79dc2514] - Operative provisions\n[bookmark: _3071ac0d-7a7f-4f84-a566-e4f8c6fe5865]Definitions \n[bookmark: _fbf6c27e-a5c8-423b-a28e-0c0cfdcfaa16]In this Schedule:\nController has the meaning given in applicable Data Protection Laws from time to time;\nData Protection Laws means any applicable law relating to the processing, privacy and/or use of Personal Data, as applicable to either party or the Services, including:\n[bookmark: _1511788371-553165]the GDPR;\n[bookmark: _1528151631-1085874327]the Data Protection Act 2018;\n[bookmark: _1511788371-770165]any laws which implement any such laws;\n[bookmark: _1511788371-777165]any laws that replace, extend, re-enact, consolidate or amend any of the foregoing; and\n[bookmark: _1511788371-784165]all guidance, guidelines, codes of practice and codes of conduct issued by any relevant Data Protection Supervisory Authority relating to such Data Protection Laws (in each case whether or not legally binding);\nData Protection Supervisory Authority means any regulator, authority or body responsible for administering Data Protection Laws;\nData Subject has the meaning given in applicable Data Protection Laws from time to time;\nGDPR means the General Data Protection Regulation, Regulation (EU) 2016/679;\nInternational Organisation has the meaning given in applicable Data Protection Laws from time to time;\nPersonal Data has the meaning given in applicable Data Protection Laws from time to time;\nPersonal Data Breach has the meaning given in applicable Data Protection Laws from time to time;\nprocessing has the meaning given in applicable Data Protection Laws from time to time (and related expressions, including process, processing, processed, and processes shall be construed accordingly);\nProcessor has the meaning given in applicable Data Protection Laws from time to time;\nProtected Data means Personal Data received from or on behalf of Simplyhealth, or otherwise obtained in connection with the performance of the Supplier’s obligations under this Agreement; and\nSub-Processor means any agent, subcontractor or other third party engaged by the Supplier (or by any other Sub-Processor) for carrying out any processing activities in respect of the Protected Data.\n[bookmark: _c3556315-efbc-45f3-bd30-1a528314292e]Unless otherwise expressly stated in this Agreement the Supplier’s obligations and Simplyhealth’s rights and remedies under this Schedule are cumulative with, and additional to, any other provisions of this Agreement.\n[bookmark: _1e8bb862-71f3-4fdd-a139-4b5a65ec754d]Compliance with Data Protection Laws\nThe parties agree that Simplyhealth is a Controller and that the Supplier is a Processor for the purposes of processing Protected Data pursuant to this Agreement. The Supplier shall, and shall ensure its Sub-Processors and each of the Supplier Personnel shall, at all times comply with all Data Protection Laws in connection with the processing of Protected Data and the provision of the Services and shall not by any act or omission cause Simplyhealth (or any other person) to be in breach of any of the Data Protection Laws. Nothing in this Agreement relieves the Supplier of any responsibilities or liabilities under Data Protection Laws.\n[bookmark: _a20dbd3f-1756-4351-b6f5-9e7e42ecd40e]Supplier indemnity\n[bookmark: _b95866ad-430f-49dd-8faa-e200d8fc47da]The Supplier shall indemnify and keep indemnified Simplyhealth against:\n[bookmark: _1bbe59e2-e92c-415e-b6c2-2fa8800fcf93]all losses, claims, damages, liabilities, fines, interest, penalties, costs, charges, sanctions, expenses, compensation paid to Data Subjects (including compensation to protect goodwill and ex gratia payments), demands and legal and other professional costs (calculated on a full indemnity basis and in each case whether or not arising from any investigation by, or imposed by, a Data Protection Supervisory Authority) arising out of or in connection with any breach by the Supplier of its obligations under this Schedule; and\n[bookmark: _3112be0b-1a58-4d68-a52f-9c4d21252e28]all amounts paid or payable by Simplyhealth to a third party which would not have been paid or payable if the Supplier’s breach of this Schedule had not occurred.\n[bookmark: _0e833d90-4d26-46ae-9103-e9df781ac286]Instructions\nThe Supplier shall only process (and shall ensure Supplier Personnel only process) the Protected Data in accordance with Section 1 of Schedule 3, Part 2 of this Schedule, this Agreement and Simplyhealth’s written instructions from time to time (including when making any transfer to which paragraph 9 relates) except where otherwise required by applicable law (and in such a case shall inform Simplyhealth of that legal requirement before processing, unless applicable law prevents it doing so on important grounds of public interest). The Supplier shall immediately inform Simplyhealth if any instruction relating to the Protected Data infringes or may infringe any Data Protection Law.\n[bookmark: _db1d272f-01fd-4524-8df1-85d3c5889a03]Security\nThe Supplier shall at all times implement and maintain appropriate technical and organisational measures to protect Protected Data against accidental, unauthorised or unlawful destruction, loss, alteration, disclosure or access. Such technical and organisational measures shall be at least equivalent to the technical and organisational measures set out in Section 2 of Schedule 3, Part 2 of this Schedule and shall reflect the nature of the Protected Data.\n[bookmark: _860a5644-3d8c-4505-a9f6-7cfcb2281691]Sub-processing and personnel\n[bookmark: _97033394-eac7-409a-863d-0b3f4d3d1d68]The Supplier shall not permit any processing of Protected Data by any agent, subcontractor or other third party (except its own employees that are subject to an enforceable obligation of confidence with regards to the Protected Data) without the prior specific written authorisation of that Sub-Processor by Simplyhealth and only then subject to such conditions as Simplyhealth may require.\tComment by Sacha Tomey: Microsoft (Azure) effectively is the subprocessor.  I guess in theory we need them to provide written authority that’s okay?\tComment by Nick Baladi: Microsoft will be a sub-processor and simplyhealth agree this is acceptable.Or just get wriiten authority that MS is subprocessor\n[bookmark: _5c387a5d-c3a8-4da7-9f9f-2dbd6f7e141f]The Supplier shall ensure that access to Protected Data is limited to the authorised persons who need access to it to supply the Services.\n[bookmark: _9560b2bf-83da-457f-a44b-797df3d93685]The Supplier shall prior to the relevant Sub-Processor carrying out any processing activities in respect of the Protected Data, appoint each Sub-Processor under a binding written contract containing the same obligations as under this Schedule in respect of Protected Data that (without prejudice to, or limitation of, the above):\n[bookmark: _d7d0eb82-ab99-46ef-8b3c-eb98c1f552f0]includes providing sufficient guarantees to implement appropriate technical and organisational measures in such a manner that the processing of the Protected Data will meet the requirements of all Data Protection Laws; and\n[bookmark: _7c395520-1bf6-47f3-a214-cc430df26f9d]is enforceable by the Supplier,\nand ensure each such Sub-Processor complies with all such obligations.\n[bookmark: _57c6d8ca-531d-4cf7-8d4e-fbc3b0f9dec4]The Supplier shall remain fully liable to Simplyhealth under this Agreement for all the acts and omissions of each Sub-Processor and each of the Supplier Personnel as if they were its own.\n[bookmark: _1fb08890-48dc-4e30-b5d4-85fbb8ffafb1]The Supplier shall ensure that all persons authorised by the Supplier or any Sub-Processor to process Protected Data are reliable and:\n[bookmark: _68b9c401-d735-4193-a399-a5efc64c87ab]adequately trained on compliance with this Schedule as applicable to the processing;\n[bookmark: _4e17b570-99dd-4dd2-8fa4-33ba0b7e66dc]informed of the confidential nature of the Protected Data and that they must not disclose Protected Data;\n[bookmark: _5eaffd93-0bdc-4d80-82d7-8af3cd33f37e]subject to a binding and enforceable written contractual obligation to keep the Protected Data confidential; and\n[bookmark: _911f9164-6d79-4290-8bff-6f495c2e5f02]provide relevant details and a copy of each agreement with a Sub-Processor to Simplyhealth on request.\n[bookmark: _35273e1f-936b-49b1-b9d1-ca83e92c08b3]Assistance\n[bookmark: _df8afbac-4f80-4fb4-97b1-cecfd357998a]The Supplier shall (at its own cost and expense) promptly provide such information and assistance (including by taking all appropriate technical and organisational measures) as Simplyhealth may require in relation to the fulfilment of Simplyhealth’s obligations to respond to requests for exercising the Data Subjects’ rights under Chapter III of the GDPR (and any similar obligations under applicable Data Protection Laws).\tComment by Sacha Tomey: We shouldn’t absorb costs for providing this information.  In reality Simply will use our system to do this themselves but with this clause they might choose to ask us.  Not keen, Suggest the cost/expense bit is removed.\tComment by Nick Baladi: Delete this section – while we will help out with GDPR requests we cannot undertake it at our own expense.\n[bookmark: _ff8322c1-027c-416a-804a-4f7543652fec]The Supplier shall (at its own cost and expense) provide such information, co-operation and other assistance to Simplyhealth as Simplyhealth reasonably requires (taking into account the nature of processing and the information available to the Supplier) to ensure compliance with Simplyhealth’s obligations under Data Protection Laws, including with respect to:\tComment by Sacha Tomey: Definiately As above.  (c) almost suggests we should absorb getting stuff agreed in advance.  Almost like a tda.\tComment by Nick Baladi: Can we delete cost and expense. For same reasons\n[bookmark: _8dfeac06-3944-4823-a870-5ca3e4bad253]security of processing;\n[bookmark: _a202b15f-17dc-4729-9d16-a486acd62bb4]data protection impact assessments (as such term is defined in Data Protection Laws);\n[bookmark: _28bb6830-38b0-4e5c-862d-9ee808d79ce8]prior consultation with a Data Protection Supervisory Authority regarding high risk processing; and\n[bookmark: _b88b6696-9d63-46af-9b0b-9d917dc4f362]any remedial action and/or notifications to be taken in response to any Personal Data Breach and/or any complaint or request relating to either party's obligations under Data Protection Laws relevant to this Agreement, including (subject in each case to Simplyhealth’s prior written authorisation) regarding any notification of the Personal Data Breach to Data Protection Supervisory Authorities and/or communication to any affected Data Subjects.\n[bookmark: _54ab924e-f264-4ccb-be27-9f364c0114f1]Data subject requests\nThe Supplier shall (at no cost to Simplyhealth) record and refer all requests and communications received from Data Subjects or any Data Protection Supervisory Authority to Simplyhealth which relate (or which may relate) to any Protected Data promptly (and in any event within three (3) days of receipt) and shall not respond to any without Simplyhealth’s express written approval and strictly in accordance with Simplyhealth’s instructions unless and to the extent required by law.\n[bookmark: _6df80568-c931-4023-8f26-3dc429c5cb93]International transfers\nThe Supplier shall not process and/or transfer, or otherwise directly or indirectly disclose, any Protected Data in or to countries outside the United Kingdom or to any International Organisation without the prior written consent of Simplyhealth (which may be refused or granted subject to such conditions as Simplyhealth deems necessary).\n[bookmark: _b2379b19-68ba-408e-8a1b-771ed8eb1023]Records\nThe Supplier shall maintain complete, accurate and up to date written records of all categories of processing activities carried out on behalf of Simplyhealth. Such records shall include all information necessary to demonstrate its and Simplyhealth’s compliance with this Schedule, the information referred to in Articles 30(1) and 30(2) of the GDPR and such other information as Simplyhealth may reasonably require from time to time. The Supplier shall make copies of such records available to Simplyhealth promptly (and in any event within five (5) Business Days) on request from time to time.\n[bookmark: _b3147dff-5157-421d-9f55-936c76f27b1d]Audit\nThe Supplier shall (and shall ensure all Sub-Processors shall) promptly make available to Simplyhealth (at the Supplier’s cost) such information as is reasonably required to demonstrate the Supplier’s and Simplyhealth’s compliance with their respective obligations under this Schedule and the Data Protection Laws, and allow for, permit and contribute to audits, including inspections, by Simplyhealth (or another auditor mandated by Simplyhealth) for this purpose at Simplyhealth’s request from time to time. The Supplier shall provide (or procure) access to all relevant premises, systems, personnel and records during normal business hours for the purposes of each such audit or inspection upon reasonable prior notice (not being more than two Business Days) and provide and procure all further reasonable co-operation, access and assistance in relation to any such audit or inspection.\n[bookmark: _6b5ff442-863c-475f-9edb-ed4e631a54e1]Breach\n[bookmark: _5b877fc3-0284-4bdc-a3a2-2db1f3dfe6af]The Supplier shall promptly (and in any event within 24 hours) notify Simplyhealth if it (or any of its Sub-Processors or the Supplier Personnel) suspects or becomes aware of any suspected, actual or threatened occurrence of any Personal Data Breach in respect of any Protected Data.\n[bookmark: _4a2c7cdb-82ab-4e99-bab6-2a37d0c65c17]The Supplier shall promptly (and in any event within 24 hours) provide all information as Simplyhealth requires to report the circumstances referred to in paragraph 12.1 (above) to a Data Protection Supervisory Authority and to notify affected Data Subjects under Data Protection Laws.\n[bookmark: _66b2c2bd-d8cd-4733-a459-510e2e2a642e]Deletion/return\n[bookmark: _76729ebf-7a50-4648-b145-b346787e0f49]The Supplier shall (and shall ensure that each of the Sub-Processors and Supplier Personnel shall)  without delay (and in any event within 3 days), at Simplyhealth’s written request, either securely delete or securely return all the Protected Data to Simplyhealth in such form as Simplyhealth reasonably requests after the earlier of:\n[bookmark: _d94f3c59-10d6-4ce0-b141-29d1c80ca81e]the end of the provision of the relevant Services related to processing of such Protected Data; or\n[bookmark: _e0a1170a-80d8-47f7-bbc5-0f2a59b2bcd8]once processing by the Supplier of any Protected Data is no longer required for the purpose of the Supplier’s performance of its relevant obligations under this Agreement,\nand securely delete existing copies (except to the extent that storage of any such data is required by applicable law and, if so, the Supplier shall inform Simplyhealth of any such requirement).\n[bookmark: _7346b27e-1c94-4c9c-848a-ae4f42be4908]Survival\nThis Schedule shall survive termination or expiry of this Agreement for any reason.\tComment by Sacha Tomey: Our GDPR obligations should end when the agreement ends.\tComment by Nick Baladi: Can we delete this section as the obligation would end when the agreement ends. We cant sign up to this being open ended.\n[bookmark: _33fdc641-606c-48ca-9934-4540d656df1f]Cost\nThe Supplier shall perform all its obligations under this Schedule at no cost to Simplyhealth.\tComment by Sacha Tomey: I don’t like this either – we can’t quote for unknown unknowns at this stage so we need to oush back on this too.\tComment by Nick Baladi: Can we delete this please – See 7.1 and 7.2\n\n\n[bookmark: _Toc256000057][bookmark: _Toc26196127][bookmark: _03c6f98d-40c3-4ac0-8223-783f93e87ced] - Data processing and security details\n[bookmark: _c118d9fb-27ce-4806-8513-340c29eb97cc]Section 1—Data processing details\n\nProcessing of the Protected Data by the Supplier under this Agreement shall be for the subject-matter, duration, nature and purposes and involve the types of Personal Data and categories of Data Subjects set out in this Section 1 of Schedule 3, Part 2.\n[bookmark: _e445e7ad-cd46-4418-89b8-b3015c6d71c4]Subject-matter of processing:\nThe Supplier shall provide (i) the Services as set out in Schedule 1 of this Agreement or (ii) the Services supplied to Simplyhealth by the Supplier pursuant to a Purchase Order.\n[bookmark: _777d7f07-1741-40c1-94e7-f0c9338c82f1]Duration of the processing:\n[bookmark: _Toc26196128]The processing will commence on the Commencement Date of this Agreement and shall continue until 31st December 2020, unless or until either party terminates the Agreement earlier in accordance with clause 2.1.\n[bookmark: _7987ec3b-a77c-4bc4-bd4d-95aef12d557b]Nature and purpose of the processing:\nThe Supplier shall assist Simplyhealth with the building of the architectural framework to pull data from source systems to the data warehouse. This may involve the processing of Personal Data. \n[bookmark: _96788494-fd6d-4a62-8a1a-bfede5aa2a8c]Type of Personal Data:\nThe Protected Data processed may concern the following categories of data:\n[bookmark: _7f8fe25d-1439-4227-9ce1-25b4761b7343]Details such as a Data Subject’s name, date of birth, gender, address, email address, telephone number, employer name, employee ID, employment and pensionable service status and periods, dates of absence, employment grade, employee performance, job title, salary and remuneration arrangements, nature and details of current and historic pension arrangements, pension amounts, pension contributions, employee benefit details, insurance cover, marital status, beneficiary details, bank details, national insurance number/national identification number/social security number, underwriting status, business travel information, educational background, passport number, driving licence number, details of power of attorney, psychometric test results, number of dependents/beneficiaries and/or ill-health status. \n\nCategories of Data Subjects:\nThe categories of Data Subject shall consist of previous and current customers.\n[bookmark: _76593abd-ed9c-4669-90ba-797510a1ed68]Specific processing instructions:\n[bookmark: _Toc26196129]The Supplier shall comply with the latest Information Security Questionnaire. \n[bookmark: _526cbc87-0c8d-4385-9ef2-cdfdab766a02]Section 2—Minimum technical and organisational security measures\n\n[bookmark: _b3ff3c8d-feea-4cfe-b96b-59297fb8cfac]Without prejudice to its other obligations, the Supplier shall implement and maintain at least the following technical and organisational security measures to protect the Protected Data:\n[bookmark: _e0ae1a15-433d-4609-ba68-e665ee36d068]In accordance with the Data Protection Laws, taking into account the state of the art, the costs of implementation and the nature, scope, context and purposes of the processing of the Protected Data to be carried out under or in connection with this Agreement, as well as the risks of varying likelihood and severity for the rights and freedoms of natural persons and the risks that are presented by the processing, especially from accidental or unlawful destruction, loss, alteration, unauthorised disclosure of, or access to the Protected Data transmitted, stored or otherwise processed, the Supplier shall implement appropriate technical and organisational security measures appropriate to the risk, including as appropriate those matters mentioned in Articles 32(1)(a) to 32(1)(d) (inclusive) of the GDPR.\n[bookmark: _9dc9b4ca-47ab-44aa-ab0f-6c3dfc590f20][bookmark: _Toc256000058][bookmark: _Toc256000120][bookmark: _Toc11940752][bookmark: _eb3d55e9-9e09-4f13-bfcc-ca77066fc45c]Without prejudice to its other obligations, the Supplier shall  comply with any and all technical and organisational data security measures as per the most up to date Information Security Questionnaire. \n\n[bookmark: _ab59eae3-cc6b-45d7-92f5-0d15eb3a2217][bookmark: _Toc256000060][bookmark: _Toc256000122][bookmark: _Toc11940754][bookmark: _Toc26196130]\n[bookmark: _Toc256000061][bookmark: _Toc26196131]Change request\n[bookmark: _Toc26196132]Change Request template:\tComment by Cassie Oakley: If Adatis would like to use their change request template then we are comfortable. Please provide a copy of the template to be included here. \n\n\tChange Request number:\n\tAgreement:\n\tEffective date of Change:\n\n\t\nInitiated by:\n\nChange requested by [Supplier OR Simplyhealth]\n\n\t\nDate of request:\n\n\t\nPeriod of validity:\n\nThis Change Request is valid for acceptance until [DATE].\n\n\t\nReason for Change Request:\n\n\t\nDescription and impact of the change (including to delivery and performance):\n\n\t\nRequired amendments to wording of agreement or schedules:\n\n\t\nAdjustment to Price(s) resulting from change:\n\n\t\nAdditional one-off charges and means of determining these (for example, fixed price basis):\n\n\t\nSupporting or additional information:\n\n\t\nSIGNED ON BEHALF OF SIMPLYHEALTH GROUP LIMITED\n\t\nSIGNED ON BEHALF OF ADATIS GROUP LIMITED\n\n\n\tSignature:\n\n\n\n\tSignature:\n\n\tName:\n\tName:\n\n\tPosition:\n\tPosition:\n\n\tDate:\n\tDate:\n\n\n\n\n\n\t1\n10-30868808-2\\326497-153\t22\n"

# COMMAND ----------

contract = nlp(la)

# COMMAND ----------

# DBTITLE 1,Check similarity of two documents
# estimate is cosine similarity using an average of word vectors.
doc.similarity(doc2)

# COMMAND ----------

# DBTITLE 1,Tokenize Sentence 
we = []
for n,s in enumerate(contract.sents):
  we.append(f'{n}:{s}')
we

# COMMAND ----------

# DBTITLE 1,Test LUIS intention prediction model on sentences
import json

import requests

url = 'https://westeurope.api.cognitive.microsoft.com/luis/v2.0/apps/97db7335-0514-4624-a7d0-179d1386a6b3?verbose=true&timezoneOffset=0&subscription-key=05fc99ce0a564c78bb93c3d87a68bae4&q='


payload =we[800]

  

#'2.1	This Agreement will come into force on the Effective Date and, unless earlier terminated in accordance with clause 10, will expire 6 months after the expiry or termination (however arising) of the exit period of the last Call-Off Contract (the “Term”). '

#'This clause represents a risk share, you may wish to reconsider the apportionment of liability and whether recoverability of losses are likely to be hindered by the contractual limitation of liability provisions'

#'the rights of the Buyer against the Guarantor under this Deed of Guarantee are in addition to, will not be affected by and will not prejudice, any other security, guarantee, indemnity or other rights or remedies available to the Buyer'

#'A Guarantee should only be requested if the Supplier’s financial standing is not enough on its own to guarantee delivery of the Services. This is a draft form of guarantee which can be used to procure a Call Off Guarantee, and so it will need to be amended to reflect the Beneficiary’s requirements'

#'1.2.5	The party receiving the benefit of an indemnity under this Agreement will use its reasonable endeavours to mitigate its loss covered by the indemnity.'

#'this does not displace the warrenty period, so even if the object needs to be repaired or replaced it will still need a letter of guarantee'

#' a new claim has come through and it seems that the trail will be run by the adjudicator in order to dispute the resolution'

#" before you quit you have to give in your notice of cancellation otheriwse your contract will be deafulted...whatever even more random text to test the predictive power of LUIS  "

#"the costs incured by the party will be awarded to him for all the damange and goodwill, whatever just text to test the predictive power"

r = requests.post(url, data=json.dumps(payload))

print(r.text)

# COMMAND ----------

for token in doc:
    print(token.text, token.vector_norm)

# COMMAND ----------

doc.vector

# COMMAND ----------

# DBTITLE 1,Tokenize Words
# for s in doc:
#   print(s.text)
  
ab = [s.text for s in doc]  
ab


# COMMAND ----------

# DBTITLE 1,Detect & Display Entities 
doc = nlp(pdf["Description"][3])
displayHTML(spacy.displacy.render(doc, style='ent'))

# COMMAND ----------

# DBTITLE 1,Part-of-speech tagging 
# for s in doc:
#   print(s.text,s.pos_)
  
for i in doc:
    print(i,"=>",i.pos_)  

# COMMAND ----------

# DBTITLE 1,Visualise Word dependencies 

displayHTML(spacy.displacy.render(doc, style='dep'))

# COMMAND ----------

# DBTITLE 1,Most common Nouns in text
nouns = [ token.text for token in doc if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN']
word_freq = Counter(nouns)
common_nouns = word_freq.most_common(10)
print(common_nouns)

# COMMAND ----------

# DBTITLE 1,Most common Verbs in text
verbs = [ token.text for token in doc if token.is_punct !=True and token.pos_ == 'VERB']
print(Counter(verbs).most_common(10))

# COMMAND ----------

# DBTITLE 1,Functions that cover/redact names or locations
# Function to Sanitize/Redact Names
def sanitize_names(text):
    doc = nlp(text)
    redacted_sentences = []
    for ent in doc.ents:
        ent.merge()
    for token in doc:
        if token.ent_type_ == 'PERSON':
            redacted_sentences.append("[REDACTED]")
        else:
            redacted_sentences.append(token.string)
    return "".join(redacted_sentences)
  
  

# Redaction of Location/GPE
def sanitize_locations(text):
    doc = nlp(text)
    redacted_sentences = []
    for ent in doc.ents:
        ent.merge()
    for token in doc:
        if token.ent_type_ == 'GPE':
            redacted_sentences.append("[REDACTED]")
        else:
            redacted_sentences.append(token.string)
    return "".join(redacted_sentences)
  
  
  
  
  
sanitize_locations(ksdf['Description'][3])
sanitize_names(ksdf['Description'][3])

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

import string

punctuations = string.punctuation
stopwords = list(STOP_WORDS)

# COMMAND ----------

review = str(" ".join([i.lemma_ for i in doc]))


# COMMAND ----------

# Parser for reviews
parser = English()
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

# COMMAND ----------

from tqdm import tqdm
tqdm.pandas()
pdf["processed_description"] = pdf["Description"].progress_apply(spacy_tokenizer)

# COMMAND ----------

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(pdf["processed_description"])

# COMMAND ----------

NUM_TOPICS = 45

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

# COMMAND ----------

nmf = NMF(n_components=NUM_TOPICS)
data_nmf = nmf.fit_transform(data_vectorized) 

# COMMAND ----------

# Latent Semantic Indexing Model using Truncated SVD
lsi = TruncatedSVD(n_components=NUM_TOPICS)
data_lsi = lsi.fit_transform(data_vectorized)

# COMMAND ----------

# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]]) 

# COMMAND ----------

# Keywords for topics clustered by Latent Dirichlet Allocation
print("LDA Model:")
selected_topics(lda, vectorizer)

# COMMAND ----------

# Keywords for topics clustered by Latent Semantic Indexing
print("NMF Model:")
selected_topics(nmf, vectorizer)

# COMMAND ----------

pyLDAvis.enable_notebook()
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
dash

# COMMAND ----------


print(python.__version__)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

PERSON   	         People, including fictional.
NORP    	         Nationalities or religious or political groups.
FAC     	         Buildings, airports, highways, bridges, etc.
ORG     	         Companies, agencies, institutions, etc.
GPE	                 Countries, cities, states.
LOC	                 Non-GPE locations, mountain ranges, bodies of water.
PRODUCT	             Objects, vehicles, foods, etc. (Not services.)
EVENT	             Named hurricanes, battles, wars, sports events, etc.
WORK_OF_ART	         Titles of books, songs, etc.
LAW      	         Named documents made into laws.
LANGUAGE	         Any named language.
DATE	             Absolute or relative dates or periods.
TIME	             Times smaller than a day.
PERCENT	             Percentage, including ”%“.
MONEY	             Monetary values, including unit.
QUANTITY	         Measurements, as of weight or distance.
ORDINAL  	         “first”, “second”, etc.
CARDINAL	         Numerals that do not fall under another type.

# COMMAND ----------

entPERSON  = []      
entNORP = []
entFAC = []
entORG  = []
entGPE  = []
entLOC  = []
entPRODUCT = []
entEVENT = []
entWORK_OF_ART = []
entLAW = []
entLANGUAGE = []
entDATE = []
entTIME = []
entPERCENT = []
entMONEY = []
entQUANTITY = []
entORDINAL = []
entCARDINAL = []


for text in nlp.pipe(iter(bestdf['Description'])):  
  for X in text.ents:
    if X.label_  == 'PERSON':
      entPERSON.append([(X.text)])
    elif X.label_  == 'ORG':
      entORG.append([(X.text)])
    elif X.label_  == 'NORP':
      entNORP.append([(X.text)])
    elif X.label_  == 'FAC':
      entFAC.append([(X.text)])
    elif X.label_  == 'GPE':
      entGPE.append([(X.text)])
    elif X.label_  == 'LOC':
      entLOC.append([(X.text)])
    elif X.label_  == 'PRODUCT':
      entPRODUCT.append([(X.text)])
    elif X.label_  == 'EVENT':
      entEVENT.append([(X.text)])
    elif X.label_  == 'WORK_OF_ART':
      entWORK_OF_ART.append([(X.text)])
    elif X.label_  == 'LAW':
      entLAW.append([(X.text)])
    elif X.label_  == 'LANGUAGE':
      entLANGUAGE.append([(X.text)])
    elif X.label_  == 'DATE':
      entDATE.append([(X.text)])
    elif X.label_  == 'TIME':
      entTIME.append([(X.text)])
    elif X.label_  == 'PERCENT':
      entPERCENT.append([(X.text)])
    elif X.label_  == 'MONEY':
      entMONEY.append([(X.text)])
    elif X.label_  == 'QUANTITY':
      entQUANTITY.append([(X.text)])
    elif X.label_  == 'ORDINAL':
      entORDINAL.append([(X.text)])
    elif X.label_  == 'CARDINAL':
      entCARDINAL.append([(X.text)])
        
          
      

#print([(X.text, X.label_)])


# COMMAND ----------



entPERSON  = ks.DataFrame()      
entNORP = ks.DataFrame()
entFAC = ks.DataFrame()
entORG  = ks.DataFrame()
entGPE  = ks.DataFrame()
entLOC  = ks.DataFrame()
entPRODUCT = ks.DataFrame()
entEVENT = ks.DataFrame()
entWORK_OF_ART = ks.DataFrame()
entLAW = ks.DataFrame()
entLANGUAGE = ks.DataFrame()
entDATE = ks.DataFrame()
entTIME = ks.DataFrame()
entPERCENT = ks.DataFrame()
entMONEY = ks.DataFrame()
entQUANTITY = ks.DataFrame()
entORDINAL = ks.DataFrame()
entCARDINAL = ks.DataFrame()

for text in nlp.pipe(iter(pdf['Description']), batch_size = 1000, n_threads=-1):  
  for X in text.ents:
    if X.label_  == 'PERSON':
      entPERSON.append([(X.text)])
    elif X.label_  == 'ORG':
      entORG.append([(X.text)])
    elif X.label_  == 'NORP':
      entNORP.append([(X.text)])
    elif X.label_  == 'FAC':
      entFAC.append([(X.text)])
    elif X.label_  == 'GPE':
      entGPE.append([(X.text)])
    elif X.label_  == 'LOC':
      entLOC.append([(X.text)])
    elif X.label_  == 'PRODUCT':
      entPRODUCT.append([(X.text)])
    elif X.label_  == 'EVENT':
      entEVENT.append([(X.text)])
    elif X.label_  == 'WORK_OF_ART':
      entWORK_OF_ART.append([(X.text)])
    elif X.label_  == 'LAW':
      entLAW.append([(X.text)])
    elif X.label_  == 'LANGUAGE':
      entLANGUAGE.append([(X.text)])
    elif X.label_  == 'DATE':
      entDATE.append([(X.text)])
    elif X.label_  == 'TIME':
      entTIME.append([(X.text)])
    elif X.label_  == 'PERCENT':
      entPERCENT.append([(X.text)])
    elif X.label_  == 'MONEY':
      entMONEY.append([(X.text)])
    elif X.label_  == 'QUANTITY':
      entQUANTITY.append([(X.text)])
    elif X.label_  == 'ORDINAL':
      entORDINAL.append(X.text)
    elif X.label_  == 'CARDINAL':
      entCARDINAL.append(X.text)
        
#print([(X.text, X.label_)])


# COMMAND ----------

entPERSO  = pd.DataFrame(entPERSON, columns=['PERSON'])      
entNOR = pd.DataFrame(entNORP, columns=['NORP'])
entFA = pd.DataFrame(entFAC, columns=['FAC'])
entOR  = pd.DataFrame(entORG, columns=['ORG'])
entGP  = pd.DataFrame(entGPE, columns=['GPE'])
entLO  = pd.DataFrame(entLOC, columns=['LOC'])
entPRODUC = pd.DataFrame(entPRODUCT, columns=['PRODUCT'])
entEVEN = pd.DataFrame(entEVENT, columns=['EVENT'])
entWORK_OF_AR = pd.DataFrame(entWORK_OF_ART, columns=['WORK_OF_ART'])
entLA = pd.DataFrame(entLAW, columns=['LAW'])
entLANGUAG = pd.DataFrame(entLANGUAGE, columns=['LANGUAGE'])
entDAT = pd.DataFrame(entDATE, columns=['DATE'])
entTIM = pd.DataFrame(entTIME, columns=['TIME'])
entPERCEN = pd.DataFrame(entPERCENT, columns=['PERCENT'])
entMONE = pd.DataFrame(entMONEY, columns=['MONEY'])
entQUANTIT = pd.DataFrame(entQUANTITY, columns=['QUANTITY'])
entORDINA = pd.DataFrame(entORDINAL, columns=['ORDINAL'])
entCARDINA = pd.DataFrame(entCARDINAL, columns=['CARDINAL'])



entPERSO.drop_duplicates(keep="first", inplace=True)
entNOR.drop_duplicates(keep="first", inplace=True)
entFA.drop_duplicates(keep="first", inplace=True)
entOR.drop_duplicates(keep="first", inplace=True)  
entGP.drop_duplicates(keep="first", inplace=True)  
entLO.drop_duplicates(keep="first", inplace=True) 
entPRODUC.drop_duplicates(keep="first", inplace=True) 
entEVEN.drop_duplicates(keep="first", inplace=True) 
entWORK_OF_AR.drop_duplicates(keep="first", inplace=True)
entLA.drop_duplicates(keep="first", inplace=True) 
entLANGUAG.drop_duplicates(keep="first", inplace=True)
entDAT.drop_duplicates(keep="first", inplace=True)
entTIM.drop_duplicates(keep="first", inplace=True) 
entPERCEN.drop_duplicates(keep="first", inplace=True) 
entMONE.drop_duplicates(keep="first", inplace=True) 
entQUANTIT.drop_duplicates(keep="first", inplace=True)
entORDINA.drop_duplicates(keep="first", inplace=True) 
entCARDINA.drop_duplicates(keep="first", inplace=True)



entPERSO  = entPERSO[~(entPERSO['PERSON'].str.len() < 5)]
entNOR = entNOR[~(entNOR['NORP'].str.len() < 5)]
entFA = entFA[~(entFA['FAC'].str.len() < 5)]
entOR  = entOR[~(entOR['ORG'].str.len() < 5)]
entGP  = entGP[~(entGP['GPE'].str.len() < 5)]
entLO  = entLO[~(entLO['LOC'].str.len() < 5)]
entPRODUC = entPRODUC[~(entPRODUC['PRODUCT'].str.len() < 5)]
entEVEN = entEVEN[~(entEVEN['EVENT'].str.len() < 5)]
entWORK_OF_AR = entWORK_OF_AR[~(entWORK_OF_AR['WORK_OF_ART'].str.len() < 5)]
entLA = entLA[~(entLA['LAW'].str.len() < 5)]
entLANGUAG = entLANGUAG[~(entLANGUAG['LANGUAGE'].str.len() < 5)]
#entDAT = entDAT[~(entDAT['DATE'].str.len() < 5)]
#entTIM = entTIM[~(entTIM['TIME'].str.len() < 5)]
#entPERCEN = entPERCEN[~(entPERCEN['PERCENT'].str.len() < 5)]
#entMONE = entMONE[~(entMONE['MONEY'].str.len() < 15)]
#entQUANTIT= entQUANTIT[~(entQUANTIT['QUANTITY'].str.len() < 5)]
#entORDINA = entORDINA[~(entORDINA['ORDINAL'].str.len() < 5)]
#entCARDINA=  entCARDINA[~(entCARDINA['CARDINAL'].str.len() < 5)]

# entPERSO  = 
# entNOR = 
# entFA = 
# entOR  =    
# entGP  = 
# entLO  = 
# entPRODUC 
# entEVEN = 
# entWORK_OF_A
# entLA = 
# entLANGUAG =
# entDAT = 
# entTIM = 
# entPERCEN =
# entMONE = 
# entQUANTIT=
# entORDINA =
# entCARDINA= 



#'ORDINAL', 'QUANTITY', 'MONEY', 'PERCENT', 'TIME', 'DATE', 'LANGUAGE', 'LAW', 'WORK_OF_ART', 'EVENT', 'PRODUCT', 'LOC', 'GPE', 'ORG', 'FAC', 'NORP', 'PERSON'

#error if using Koalas
#ArrowInvalid: Could not convert ('Peyer', 'PERSON') with type tuple: did not recognize Python value type when inferring an Arrow data type

# COMMAND ----------

fullEnt = pd.concat([entPERSO, entNOR, entFA, entOR, entGP, entLO, entPRODUC, entEVEN, entWORK_OF_AR, entLA, entLANGUAG, entDAT, entTIM, entPERCEN, entMONE, entQUANTIT, entORDINA, entCARDINA],axis=1)
fullEnt


# COMMAND ----------

pd.set_option('display.max_colwidth', -1)

bestdf.iloc[2]

# COMMAND ----------

# DBTITLE 1,Get entity list of whole DF
entities = []
for text in nlp.pipe(iter(bestdf['Description']), batch_size = 1000, n_threads=-1):  
  entities.append([(X.text, X.label_) for X in text.ents])
  print([(X.text, X.label_) for X in text.ents])


# COMMAND ----------

#index = range(2)
index=range(0,13911)
full_df = pd.DataFrame(index = index, columns=['DOC NR','CARDINAL', 'ORDINAL', 'QUANTITY', 'MONEY', 'PERCENT', 'TIME', 'DATE', 'LANGUAGE', 'LAW', 'WORK_OF_ART', 'EVENT', 'PRODUCT', 'LOC', 'GPE', 'ORG', 'FAC', 'NORP', 'PERSON'])


for index, doc in enumerate(entities):
    for entity, entity_type in doc:
        if pd.isna(full_df.at[index, entity_type]):
            full_df.at[index, entity_type] = entity
        else:
            full_df.at[index, entity_type] = full_df.at[index, entity_type] + ", {}".format(entity)

# COMMAND ----------

full_df

# COMMAND ----------

cleanDFA = full_df.replace(np.nan, 'Unknown', regex=True)

# COMMAND ----------

cleanDFA

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

