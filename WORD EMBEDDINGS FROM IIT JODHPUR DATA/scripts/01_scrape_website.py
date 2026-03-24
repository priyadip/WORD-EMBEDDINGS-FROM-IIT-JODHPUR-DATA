"""
IIT Jodhpur Website Scraper
=============================================
Crawls the IIT Jodhpur official website (https://www.iitj.ac.in) and its
relevant subpages (departments, academic programs, research, faculty profiles,
announcements). Also downloads PDFs found on the pages (e.g. academic
regulations, syllabi, newsletters).

Sources covered:
  - IIT Jodhpur official website pages
  - Academic regulation documents (PDFs) 
  - Faculty profile pages
  - Course syllabus / department pages
  - Institute announcements / circulars

Output:
  - data/raw/text/  : one .txt file per scraped web page (English text only)
  - data/raw/pdfs/  : downloaded PDF files
  - data/raw/scrape_log.json : metadata about every scraped URL
"""

import os
import re
import sys
import time
import json
import hashlib
import requests
import unicodedata
from urllib.parse import urljoin, urlparse
from collections import deque

from bs4 import BeautifulSoup

# Force UTF-8 output so URLs with non-ASCII characters don't crash on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Configuration

BASE_URL    = "https://www.iitj.ac.in"

# Comprehensive seed URLs covering every major section of iitj.ac.in.
# The BFS crawler will follow all internal links discovered from these pages,
# so seeding every section ensures maximum coverage even if a section is not
# reachable from the homepage.
START_URLS  = [

    "https://www.iitj.ac.in/",
    "https://www.iitj.ac.in/main/en/introduction",
    "https://www.iitj.ac.in/main/en/director",
    "https://www.iitj.ac.in/main/en/chairman",
    "https://www.iitj.ac.in/main/en/news",
    "https://www.iitj.ac.in/main/en/events",
    "https://www.iitj.ac.in/main/en/all-announcement",
    "https://www.iitj.ac.in/main/en/research-highlight",
    "https://www.iitj.ac.in/main/en/important-links",
    "https://www.iitj.ac.in/main/en/contact",
    "https://www.iitj.ac.in/main/en/how-to-reach-iit-jodhpur",
    "https://www.iitj.ac.in/main/en/recruitments",
    "https://www.iitj.ac.in/main/en/intranet-page",
    "https://www.iitj.ac.in/main/en/web-policy",
    "https://www.iitj.ac.in/main/en/Help",
    "https://www.iitj.ac.in/Main/en/Acts-Statutes",
    "https://www.iitj.ac.in/Main/en/Annual-Reports-of-the-Institute",
    "https://www.iitj.ac.in/Main/en/Minutes-of-the-Meetings",
    "https://www.iitj.ac.in/Main/en/Minutes-of-the-Meetings-Senate",
    "https://www.iitj.ac.in/Main/en/Administrative-Contact",
    "https://www.iitj.ac.in/Main/en/House-Allotment-Rules-for-Employee",

    "https://www.iitj.ac.in/Office-of-Academics/en/Academic-Regulations",
    "https://www.iitj.ac.in/Office-of-Academics/en/Academics",
    "https://www.iitj.ac.in/Office-of-Academics/en/Academic-Calendar",
    "https://www.iitj.ac.in/Office-of-Academics/en/Curriculum",
    "https://www.iitj.ac.in/Office-of-Academics/en/Fee-Structure",
    "https://www.iitj.ac.in/Office-of-Academics/en/Examination-Schedule",
    "https://www.iitj.ac.in/Office-of-Academics/en/Grading-System",
    "https://www.iitj.ac.in/Office-of-Academics/en/Time-Table",
    "https://www.iitj.ac.in/Office-of-Academics/en/Course-add-drop",
    "https://www.iitj.ac.in/Office-of-Academics/en/Medals-Awards",

    # B.Tech
    "https://www.iitj.ac.in/bachelor-of-technology/en/important-dates",
    "https://www.iitj.ac.in/Bachelor-of-Technology/en/Bachelor-of-Technology",
    "https://www.iitj.ac.in/Bachelor-of-Technology/en/About-FAQs",
    # M.Tech
    "https://www.iitj.ac.in/Master-of-Technology/en/Master-of-Technology",
    "https://www.iitj.ac.in/Master-of-Technology/en/FAQs",
    # M.Sc.
    "https://www.iitj.ac.in/Master-of-Science/en/FAQs",
    "https://www.iitj.ac.in/Master-of-Science/en/Assistantship",
    # M.Sc.-M.Tech Dual
    "https://www.iitj.ac.in/M.Sc.-M.Tech.-Dual-Degree/en/FAQs",
    # M.Tech-PhD Dual
    "https://www.iitj.ac.in/M.Tech.-Ph.D.-Dual-Degree/en/M.Tech.-Ph.D.-Dual-Degree",
    "https://www.iitj.ac.in/M.Tech.-Ph.D.-Dual-Degree/en/FAQs",
    # PhD
    "https://www.iitj.ac.in/Doctor-of-Philosophy/en/Doctor-of-Philosophy",
    "https://www.iitj.ac.in/Doctor-of-Philosophy/en/FAQs",
    # Post-Doctoral
    "https://www.iitj.ac.in/Post-Doctoral-Fellows/en/PDF-Positions",
    "https://www.iitj.ac.in/Post-Doctoral-Fellows/en/FAQs",
    # MBA / Schools
    "https://www.iitj.ac.in/schools/en/cat-cut-offs",
    "https://www.iitj.ac.in/schools/en/eligibility-criteria-and-selection-process",
    "https://www.iitj.ac.in/schools/en/how-to-apply",
    "https://www.iitj.ac.in/schools/en/fee-structure-student-intake",
    "https://www.iitj.ac.in/schools/en/faq-1",
    # Admissions PG
    "https://www.iitj.ac.in/admission-postgraduate-programs/en/Advertisements",
    "https://www.iitj.ac.in/admission-postgraduate-programs/en/list-of-shortlisted-candidates",
    "https://www.iitj.ac.in/admission-postgraduate-programs/en/list-of-provisionally-selected-candidates",

    # Bioscience & Bioengineering (BSBE)
    "https://www.iitj.ac.in/bioscience-bioengineering/en/About",
    "https://www.iitj.ac.in/bioscience-bioengineering/en/People",
    "https://www.iitj.ac.in/bioscience-bioengineering/en/Research",
    "https://www.iitj.ac.in/bioscience-bioengineering/en/Research-Highlights",
    "https://www.iitj.ac.in/bioscience-bioengineering/en/Courses",
    "https://www.iitj.ac.in/bioscience-bioengineering/en/Events",
    "https://www.iitj.ac.in/bioscience-bioengineering/en/Facilities",
    "https://www.iitj.ac.in/bioscience-bioengineering",
    # Chemistry
    "https://www.iitj.ac.in/chemistry/en/About",
    "https://www.iitj.ac.in/chemistry/en/People",
    "https://www.iitj.ac.in/chemistry/en/Research",
    "https://www.iitj.ac.in/chemistry/en/Research-Highlights",
    "https://www.iitj.ac.in/chemistry/en/Courses",
    "https://www.iitj.ac.in/chemistry/en/Events",
    "https://www.iitj.ac.in/chemistry/en/Facilities",
    "https://www.iitj.ac.in/chemistry",
    # Chemical Engineering
    "https://www.iitj.ac.in/chemical-engineering/en/About",
    "https://www.iitj.ac.in/chemical-engineering/en/People",
    "https://www.iitj.ac.in/chemical-engineering/en/Research",
    "https://www.iitj.ac.in/chemical-engineering/en/Research-Highlights",
    "https://www.iitj.ac.in/chemical-engineering/en/Courses",
    "https://www.iitj.ac.in/chemical-engineering/en/Events",
    "https://www.iitj.ac.in/chemical-engineering/en/Facilities",
    "https://www.iitj.ac.in/chemical-engineering",
    # Civil & Infrastructure Engineering
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/en/About",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/en/People",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/en/Research",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/en/Research-Highlights",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/en/Courses",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/en/Events",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/en/Facilities",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering",
    # Computer Science & Engineering (CSE)
    "https://www.iitj.ac.in/computer-science-engineering/en/About",
    "https://www.iitj.ac.in/computer-science-engineering/en/People",
    "https://www.iitj.ac.in/computer-science-engineering/en/Research",
    "https://www.iitj.ac.in/computer-science-engineering/en/Research-Highlights",
    "https://www.iitj.ac.in/computer-science-engineering/en/Courses",
    "https://www.iitj.ac.in/computer-science-engineering/en/Events",
    "https://www.iitj.ac.in/computer-science-engineering/en/Facilities",
    "https://www.iitj.ac.in/computer-science-engineering",
    # Electrical Engineering (EE)
    "https://www.iitj.ac.in/electrical-engineering/en/About",
    "https://www.iitj.ac.in/electrical-engineering/en/People",
    "https://www.iitj.ac.in/electrical-engineering/en/Research",
    "https://www.iitj.ac.in/electrical-engineering/en/Research-Highlights",
    "https://www.iitj.ac.in/electrical-engineering/en/Courses",
    "https://www.iitj.ac.in/electrical-engineering/en/Events",
    "https://www.iitj.ac.in/electrical-engineering/en/Facilities",
    "https://www.iitj.ac.in/electrical-engineering",
    # Mathematics
    "https://www.iitj.ac.in/mathematics/en/About",
    "https://www.iitj.ac.in/mathematics/en/People",
    "https://www.iitj.ac.in/mathematics/en/Research",
    "https://www.iitj.ac.in/mathematics/en/Research-Highlights",
    "https://www.iitj.ac.in/mathematics/en/Courses",
    "https://www.iitj.ac.in/mathematics/en/Events",
    "https://www.iitj.ac.in/mathematics",
    # Mechanical Engineering (ME)
    "https://www.iitj.ac.in/mechanical-engineering/en/About",
    "https://www.iitj.ac.in/mechanical-engineering/en/People",
    "https://www.iitj.ac.in/mechanical-engineering/en/Research",
    "https://www.iitj.ac.in/mechanical-engineering/en/Research-Highlights",
    "https://www.iitj.ac.in/mechanical-engineering/en/Courses",
    "https://www.iitj.ac.in/mechanical-engineering/en/Events",
    "https://www.iitj.ac.in/mechanical-engineering/en/Facilities",
    "https://www.iitj.ac.in/mechanical-engineering",
    # Metallurgical & Materials Engineering (MME)
    "https://www.iitj.ac.in/metallurgical-and-materials-engineering/en/About",
    "https://www.iitj.ac.in/metallurgical-and-materials-engineering/en/People",
    "https://www.iitj.ac.in/metallurgical-and-materials-engineering/en/Research",
    "https://www.iitj.ac.in/metallurgical-and-materials-engineering/en/Research-Highlights",
    "https://www.iitj.ac.in/metallurgical-and-materials-engineering/en/Courses",
    "https://www.iitj.ac.in/metallurgical-and-materials-engineering/en/Events",
    "https://www.iitj.ac.in/metallurgical-and-materials-engineering",
    # Physics
    "https://www.iitj.ac.in/physics/en/About",
    "https://www.iitj.ac.in/physics/en/People",
    "https://www.iitj.ac.in/physics/en/Research",
    "https://www.iitj.ac.in/physics/en/Research-Highlights",
    "https://www.iitj.ac.in/physics/en/Courses",
    "https://www.iitj.ac.in/physics/en/Events",
    "https://www.iitj.ac.in/physics",

    # AI & Data Science (AIDE / SCAI)
    "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/About",
    "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/People",
    "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/Research",
    "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/Research-Highlights",
    "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/Courses",
    "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/Events",
    "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science",
    # School of Design
    "https://www.iitj.ac.in/school-of-design/en/About",
    "https://www.iitj.ac.in/school-of-design/en/People",
    "https://www.iitj.ac.in/school-of-design/en/Research",
    "https://www.iitj.ac.in/school-of-design/en/Courses",
    "https://www.iitj.ac.in/school-of-design/en/Events",
    "https://www.iitj.ac.in/school-of-design",
    # School of Liberal Arts (SOLA)
    "https://www.iitj.ac.in/school-of-liberal-arts/en/About",
    "https://www.iitj.ac.in/school-of-liberal-arts/en/People",
    "https://www.iitj.ac.in/school-of-liberal-arts/en/Research",
    "https://www.iitj.ac.in/school-of-liberal-arts/en/Courses",
    "https://www.iitj.ac.in/school-of-liberal-arts/en/Events",
    "https://www.iitj.ac.in/school-of-liberal-arts",

    # Centre for Energy Technology & Sustainable Development (CETSD)
    "https://www.iitj.ac.in/cetsd/en/About",
    "https://www.iitj.ac.in/cetsd/en/People",
    "https://www.iitj.ac.in/cetsd/en/Research",
    "https://www.iitj.ac.in/cetsd/en/Courses",
    # CETE (Electric Vehicles & Intelligent Transport)
    "https://www.iitj.ac.in/cete/en/About",
    "https://www.iitj.ac.in/cete/en/People",
    "https://www.iitj.ac.in/cete/en/Research",
    # DIA Centre of Excellence
    "https://www.iitj.ac.in/dia/en/dia",
    "https://www.iitj.ac.in/dia/en/About",
    "https://www.iitj.ac.in/dia/en/People",
    "https://www.iitj.ac.in/dia-coe/en/About",
    # IoT (Internet of Things)
    "https://www.iitj.ac.in/iot/en/About",
    "https://www.iitj.ac.in/iot/en/People",
    "https://www.iitj.ac.in/iot/en/Research",
    "https://www.iitj.ac.in/iot/en/Courses",
    # QIC (Quantum Information Centre)
    "https://www.iitj.ac.in/qic/en/About",
    "https://www.iitj.ac.in/qic/en/People",
    "https://www.iitj.ac.in/qic/en/Research",
    # SST (Smart Systems and Technology)
    "https://www.iitj.ac.in/sst/en/About",
    "https://www.iitj.ac.in/sst/en/People",
    # Digital Humanities (DH)
    "https://www.iitj.ac.in/dh/en/About",
    "https://www.iitj.ac.in/dh/en/People",
    "https://www.iitj.ac.in/dh/en/Courses",
    # Resource Management (RM)
    "https://www.iitj.ac.in/rm/en/About",
    "https://www.iitj.ac.in/rm/en/People",
    # Medical Technologies
    "https://www.iitj.ac.in/Medical-Technologies/en/About",
    "https://www.iitj.ac.in/Medical-Technologies/en/People",
    "https://www.iitj.ac.in/Medical-Technologies/en/Research",
    # Center for Technology Foresight and Policy
    "https://www.iitj.ac.in/Center-for-Technology-Foresight-and-Policy/en/About",
    "https://www.iitj.ac.in/Center-for-Technology-Foresight-and-Policy/en/People",
    # CDH
    "https://www.iitj.ac.in/cdh/en/About",
    "https://www.iitj.ac.in/cdh/en/People",

    "https://www.iitj.ac.in/People?dept=bioscience-bioengineering",
    "https://www.iitj.ac.in/People?dept=Chemistry",
    "https://www.iitj.ac.in/People?dept=Chemical-Engineering",
    "https://www.iitj.ac.in/People?dept=Civil-and-Infrastructure-Engineering",
    "https://www.iitj.ac.in/People?dept=Computer-Science-Engineering",
    "https://www.iitj.ac.in/People?dept=Electrical-Engineering",
    "https://www.iitj.ac.in/People?dept=Mathematics",
    "https://www.iitj.ac.in/People/?dept=mechanical-engineering",
    "https://www.iitj.ac.in/People?dept=Metallurgical-and-Materials-Engineering",
    "https://www.iitj.ac.in/People?dept=Physics",
    "https://www.iitj.ac.in/People/?dept=school-of-artificial-intelligence-data-science",
    "https://www.iitj.ac.in/People/?dept=school-of-design",
    "https://www.iitj.ac.in/People/?dept=school-of-liberal-arts",
    "https://www.iitj.ac.in/People/?dept=schools",
    "https://www.iitj.ac.in/People?dept=cetsd",
    "https://www.iitj.ac.in/People?dept=dia",
    "https://www.iitj.ac.in/People/?dept=iot",
    "https://www.iitj.ac.in/People?dept=qic",
    "https://www.iitj.ac.in/People?dept=sst",
    "https://www.iitj.ac.in/People?dept=dh",
    "https://www.iitj.ac.in/People?dept=rm",
    "https://www.iitj.ac.in/People?dept=cdh",
    "https://www.iitj.ac.in/People?dept=Medical-Technologies",
    "https://www.iitj.ac.in/People?dept=Center-for-Technology-Foresight-and-Policy",
    "https://www.iitj.ac.in/People/?dept=crf",
    "https://www.iitj.ac.in/People?dept=aiot-fab-facility",

    "https://www.iitj.ac.in/People?dept=office-of-director",
    "https://www.iitj.ac.in/People?dept=Office-of-Deputy-Director",
    "https://www.iitj.ac.in/People?dept=office-of-registrar",
    "https://www.iitj.ac.in/People?dept=Office-of-Administration",
    "https://www.iitj.ac.in/People?dept=Centre-for-Continuing-Education",
    "https://www.iitj.ac.in/People?dept=office-of-accounts",
    "https://www.iitj.ac.in/People?dept=Office-of-Academics",
    "https://www.iitj.ac.in/People?dept=office-of-students",
    "https://www.iitj.ac.in/People?dept=office-of-international-relations",
    "https://www.iitj.ac.in/People?dept=office-of-research-development",
    "https://www.iitj.ac.in/People?dept=office-of-training-and-placement",
    "https://www.iitj.ac.in/People?dept=office-of-corporate-relations",
    "https://www.iitj.ac.in/People?dept=office-of-establishment-e-I",
    "https://www.iitj.ac.in/People?dept=office-of-recruitment-nf",
    "https://www.iitj.ac.in/People?dept=office-of-establishment-nf",
    "https://www.iitj.ac.in/People?dept=health-center",
    "https://www.iitj.ac.in/People?dept=legal-cell",
    "https://www.iitj.ac.in/People?dept=office-of-alumni-affairs",
    "https://www.iitj.ac.in/People?dept=office-of-executive-education",
    "https://www.iitj.ac.in/People?dept=office-of-estate",
    "https://www.iitj.ac.in/People?dept=office-of-infrastructure-engineering",
    "https://www.iitj.ac.in/People?dept=office-of-internal-audit",
    "https://www.iitj.ac.in/People?dept=office-of-stores-purchase",
    "https://www.iitj.ac.in/People?dept=rti-cell",
    "https://www.iitj.ac.in/People?dept=office-of-security-transports",
    "https://www.iitj.ac.in/People?dept=transit-accommodation-guest-house",
    "https://www.iitj.ac.in/People?dept=office-of-iitj-connect-pro",
    "https://www.iitj.ac.in/People?dept=green-cell",
    "https://www.iitj.ac.in/People?dept=hindi-cell",
    "https://www.iitj.ac.in/People?dept=manekshaw-centre",
    "https://www.iitj.ac.in/cete/en/People",

    "https://www.iitj.ac.in/office-of-students/en/Office-of-Students",
    "https://www.iitj.ac.in/office-of-students/en/campus-life",
    "https://www.iitj.ac.in/office-of-students/en/Student-Life-@-IIT-Jodhpur",
    "https://www.iitj.ac.in/office-of-students/en/Clubs-and-Activities",
    "https://www.iitj.ac.in/office-of-students/en/Sports",
    "https://www.iitj.ac.in/office-of-students/en/Hostel",
    "https://www.iitj.ac.in/office-of-students/en/Scholarships",
    "https://www.iitj.ac.in/office-of-students/en/Medical-Facilities",
    "https://www.iitj.ac.in/office-of-students/en/Gymkhana",

    "https://www.iitj.ac.in/office-of-research-development/en/About",
    "https://www.iitj.ac.in/office-of-research-development/en/project-staff-appointment",
    "https://www.iitj.ac.in/office-of-research-development/en/Sponsored-Research",
    "https://www.iitj.ac.in/office-of-research-development/en/Consultancy",
    "https://www.iitj.ac.in/office-of-research-development/en/Industry-Collaboration",
    "https://www.iitj.ac.in/office-of-research-development/en/Patent-TTO",

    "https://www.iitj.ac.in/office-of-training-and-placement/en/About",
    "https://www.iitj.ac.in/office-of-training-and-placement/en/Placement-Statistics",
    "https://www.iitj.ac.in/office-of-training-and-placement/en/Recruiters",
    "https://www.iitj.ac.in/office-of-training-and-placement/en/Internship",

    "https://www.iitj.ac.in/office-of-corporate-relations/en/Donate",
    "https://www.iitj.ac.in/office-of-corporate-relations/en/About",
    "https://www.iitj.ac.in/support-iit-jodhpur/en/support-iit-jodhpur",

    "https://www.iitj.ac.in/office-of-international-relations/en/About",
    "https://www.iitj.ac.in/office-of-international-relations/en/MoUs-and-Collaborations",
    "https://www.iitj.ac.in/office-of-international-relations/en/Exchange-Programs",
    "https://www.iitj.ac.in/office-of-international-relations/en/International-Students",

    "https://www.iitj.ac.in/library/en/library",
    "https://library.iitj.ac.in/team.html",
    "https://library.iitj.ac.in",

    "https://www.iitj.ac.in/Institute-Repository/en/Institute-Repository",
    "https://www.iitj.ac.in/institute-repository/en/nirf",
    "https://iitj.ac.in/Institute-Repository/en/NIRF",
    "https://iitj.ac.in/Institute-Repository/en/Brochure",
    "https://iitj.ac.in/Institute-Repository/en/Accessible",
    "https://iitj.ac.in/Institute-Repository/en/ARIIA-2019-20",

    "https://www.iitj.ac.in/techscape/en/Techscape",

    "https://www.iitj.ac.in/rti",

    "https://www.iitj.ac.in/Faculty-Positions/en/Faculty-Positions",
    "https://www.iitj.ac.in/Recruitment/AdvNum?type=3052",
    "https://www.iitj.ac.in/Recruitment/AdvNum?type=3052&level=54",

    "https://www.iitj.ac.in/correspondence",
    "https://www.iitj.ac.in/Correspondence/index",
    "https://www.iitj.ac.in/Correspondence/index?ar=archive",

    "https://www.iitj.ac.in/office-of-stores-purchase/en/tender-details",
    "https://www.iitj.ac.in/tenders/index?ar=archive",

    "https://www.iitj.ac.in/anti-sexual-harassment-policy/en/anti-sexual-harassment-policy",
    "https://www.iitj.ac.in/SC-and-ST-Helpdesk/en/SC-and-ST-Helpdesk",
    "https://www.iitj.ac.in/inclusivity-cell/en/About",
    "https://www.iitj.ac.in/dia/en/dia",
]

# Where to save files
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.join(SCRIPT_DIR, "..", "data", "raw")
TEXT_DIR     = os.path.join(BASE_DIR, "text")
PDF_DIR      = os.path.join(BASE_DIR, "pdfs")
LOG_PATH     = os.path.join(BASE_DIR, "scrape_log.json")

# Crawler limits
MAX_PAGES        = 8000   # max HTML pages to scrape (raised for comprehensive coverage)
MAX_PDFS         = 2000   # max PDFs to download
CRAWL_DELAY_SEC  = 0.8   # polite delay between requests (seconds)
REQUEST_TIMEOUT  = 15    # seconds per request

# Only follow links that stay within iitj.ac.in
ALLOWED_DOMAIN = "iitj.ac.in"

# Ignore these path prefixes (login, calendar widgets, etc.)
SKIP_PATH_PREFIXES = [
    "/wp-content/", "/wp-includes/", "/feed/", "/tag/",
    "/page/", "?replytocom", "#", "javascript:", "mailto:",
    "/cdn-cgi/", "/wp-json/",
]

# HTML tags whose text is purely navigation / boilerplate
BOILERPLATE_TAGS = ["nav", "header", "footer", "script", "style",
                    "noscript", "aside", "form"]

# Utility helpers

def make_session():
    """Return a requests.Session with a realistic browser User-Agent."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })
    return s


def url_key(url):
    """Return a short, filesystem-safe identifier for a URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def is_internal(url):
    """True if the URL belongs to the allowed domain."""
    parsed = urlparse(url)
    return ALLOWED_DOMAIN in parsed.netloc


def should_skip(url):
    """True if the URL should not be crawled (non-HTML resources, skip paths)."""
    low = url.lower()
    # Skip common binary / non-text extensions other than PDF
    skip_exts = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
                 ".mp4", ".avi", ".mov", ".zip", ".rar", ".gz",
                 ".css", ".js", ".xml", ".rss", ".json", ".php?")
    if any(low.endswith(ext) or ext in low for ext in skip_exts):
        return True
    # Skip undesired path prefixes
    parsed = urlparse(url)
    if any(parsed.path.startswith(p) for p in SKIP_PATH_PREFIXES):
        return True
    return False


def is_english_text(text):
    """
    Heuristic to decide whether a block of text is primarily English.
    We check what fraction of alphabetic characters are ASCII (Latin script).
    Devanagari (Hindi) and other non-Latin scripts have Unicode code points
    outside the ASCII range.
    """
    alpha_chars = [c for c in text if unicodedata.category(c).startswith("L")]
    if not alpha_chars:
        return False
    ascii_alpha = sum(1 for c in alpha_chars if ord(c) < 128)
    return (ascii_alpha / len(alpha_chars)) > 0.85   # >85 % Latin → English


def extract_english_text(soup):
    """
    Extract clean English text from a BeautifulSoup object.
    Steps:
      1. Remove boilerplate tags (nav, footer, script, style …)
      2. Get visible text from <main>, <article>, <section>, <div> blocks
      3. Filter out non-English paragraphs / sentences
      4. Return joined English text
    """
    # Step 1 – remove navigation / boilerplate elements
    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()

    # Step 2 – prefer content areas; fall back to entire body
    content_area = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id=re.compile(r"content|main|body", re.I)) or
        soup.find(class_=re.compile(r"content|main|body", re.I)) or
        soup.find("body") or
        soup
    )

    # Step 3 – collect paragraph-level text, keep only English paragraphs
    english_chunks = []
    for element in content_area.find_all(
        ["p", "li", "h1", "h2", "h3", "h4", "h5", "td", "span", "div"],
        recursive=True
    ):
        # Use the direct text of each element (not nested children) to avoid dups
        raw = element.get_text(separator=" ", strip=True)
        if len(raw) < 20:          # skip very short snippets
            continue
        if is_english_text(raw):
            english_chunks.append(raw)

    return "\n".join(english_chunks)


def normalise_url(url, base):
    """Resolve relative URLs and strip fragments/trailing slashes."""
    url = urljoin(base, url).split("#")[0].rstrip("/")
    return url


def collect_links(soup, page_url):
    """Return all internal links found on the page."""
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        full = normalise_url(href, page_url)
        if is_internal(full) and not should_skip(full):
            links.append(full)
    return links


def collect_pdf_links(soup, page_url):
    """Return absolute URLs of all PDF links found on the page."""
    pdfs = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip().lower()
        if href.endswith(".pdf"):
            full = normalise_url(a["href"].strip(), page_url)
            pdfs.append(full)
    return pdfs


# Core crawl functions

def scrape_page(session, url):
    """
    Download a single page and return (english_text, links, pdf_links).
    Returns (None, [], []) on failure.
    """
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        # Only process HTML responses
        content_type = resp.headers.get("Content-Type", "")
        if "html" not in content_type:
            return None, [], []
        soup = BeautifulSoup(resp.text, "lxml")
        text  = extract_english_text(soup)
        links = collect_links(soup, url)
        pdfs  = collect_pdf_links(soup, url)
        return text, links, pdfs
    except Exception as exc:
        print(f"  [WARN] Could not scrape {url}: {exc}")
        return None, [], []


def download_pdf(session, url, pdf_dir):
    """
    Download a PDF file and save it to pdf_dir.
    Returns the local file path on success, None on failure.
    """
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        resp.raise_for_status()
        basename = os.path.basename(urlparse(url).path)
        stem, ext = os.path.splitext(basename)
        filename = url_key(url) + "_" + stem
        # Sanitise and truncate stem only, then re-attach extension
        filename = re.sub(r"[^\w.\-]", "_", filename)[:96] + ext
        filepath = os.path.join(pdf_dir, filename)
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  [PDF] Downloaded: {filename}")
        return filepath
    except Exception as exc:
        print(f"  [WARN] Could not download PDF {url}: {exc}")
        return None


def run_crawler():
    """
    BFS crawler starting from START_URLS.
    Saves English text of each page to TEXT_DIR.
    Downloads all PDF links to PDF_DIR.
    Writes a JSON log to LOG_PATH.
    """
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR,  exist_ok=True)

    session     = make_session()
    visited     = set()
    pdf_visited = set()
    queue       = deque(START_URLS)
    log         = []

    pages_saved = 0
    pdfs_saved  = 0

    print(f"Starting crawler. Max pages={MAX_PAGES}, Max PDFs={MAX_PDFS}")
    print(f"Output -> text: {TEXT_DIR}")
    print(f"       -> pdfs: {PDF_DIR}\n")

    while queue and pages_saved < MAX_PAGES:
        url = queue.popleft()

        # Normalise & skip already-visited
        url = url.rstrip("/")
        if url in visited:
            continue
        visited.add(url)

        print(f"[{pages_saved+1}/{MAX_PAGES}] Scraping: {url}")

        text, links, pdf_links = scrape_page(session, url)

        entry = {"url": url, "status": "skipped", "text_file": None, "pdfs": []}

        if text and len(text.strip()) > 100:
            filename   = url_key(url) + ".txt"
            filepath   = os.path.join(TEXT_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                # Save URL as first line for provenance
                f.write(f"SOURCE_URL: {url}\n\n")
                f.write(text)
            entry["status"]    = "saved"
            entry["text_file"] = filename
            pages_saved += 1
            print(f"  [OK] Saved text ({len(text)} chars)")
        else:
            print(f"  [SKIP] No usable English text.")

        for link in links:
            lnorm = link.rstrip("/")
            if lnorm not in visited:
                queue.append(lnorm)

        for pdf_url in pdf_links:
            if pdfs_saved >= MAX_PDFS:
                break
            if pdf_url in pdf_visited:
                continue
            pdf_visited.add(pdf_url)
            local_path = download_pdf(session, pdf_url, PDF_DIR)
            if local_path:
                entry["pdfs"].append({"url": pdf_url, "file": os.path.basename(local_path)})
                pdfs_saved += 1

        log.append(entry)

        # Polite delay between requests
        time.sleep(CRAWL_DELAY_SEC)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Crawl complete.")
    print(f"  Pages saved : {pages_saved}")
    print(f"  PDFs saved  : {pdfs_saved}")
    print(f"  Log         : {LOG_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_crawler()
