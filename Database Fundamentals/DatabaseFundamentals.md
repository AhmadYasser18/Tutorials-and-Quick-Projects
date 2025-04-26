# Database Fundamentals

Content:
- [Introduction:](#intro)
    - [Database Concepts](#concepts)
    - [Database System Main Components](#components)
    - [Database Users](#users)
    - [DBMS Architecture, Data Models](#dbms)
    - [Mappings](#mappings)
    - [DBMS other functions](#other_func)
    - [Centralized Database Environment](#envs)
    - [Distributed Database Environment](#dis_envs)

- [Entity Relationship Diagram](#ERD) 
    - [Entity Relationship Modeling](#erm)
    - [Entities and Attributes](#entities&att)

# Introduction
<a id="intro"></a>

## Database Concepts
<a id="concepts"></a>

### Limitations of File Based Systems
- Separation and isolation of data
- Duplication of data
- Program data dependence
- Incompatible file formats

**Database:** A collection of related data.

**Database Management System (DBMS):** A software package/ system to facilitate the creation and maintenance of a computerized database.

**Database System:** The DBMS software with the data itself. Sometimes, the applications are also included. (Software + Database)

## Database System Main Components
<a id="components"></a>

![alt text](image.png)

 - Database part is divided into two parts
    - Database Definition **Metadata**: set of information about the data; Table name, columns names, columns data types, columns constraints
        
    - Stored Database: contains data itself

Database System advantages:  
- Controlling redundancy
- Restricting Unauthorized Access
- Sharing Data
- Enforcing Integrity Constraints
- Inconsistency can be avoided
- Providing Backup and Recovery

Disadvantages:  
- Needs expertise to use
- DBMS is expensive
- Some DBMS may be incompatible with others

## Database Users
<a id="users"></a>
Database Creation Cycle:  
1. Analysis and requirement gathering --> System Analyst
2. Database Design --> Database Designer
3. Implementation --> (DBA) Database Adminstrator
4. Application development --> Application programmer

## DBMS Architecture, Data Models
<a id="dbms"></a>

### DBMS Architecture  
DBMS consists of 3 schema architecture:
1. External Schema:  
    - More than one exists: External schema 1, External schema 2, External schema 3
    - contains data seen by all type of users
2. Conceptual Schema:  
    - schema that contains all tables and all relations between the data 
3. Physical Schema:
    - explains the allocation of the data on the disk

The three schemas are placed separately in order to achieve data independence.  
**Data independence:** the capacity of changing a schema without affecting others at a higher level.

### Data Models
1. Logical/Conceptual model: 
    - represents the conceptual schema containing the full design of the database schema
2. Physical model:
    - explains how data is stored in the disk and its access paths

## Mappings
<a id="mapping"></a>

**Mappings:** is the proccess of transforming requests and results between levels

## DBMS other functions
<a id="other_func"></a>
In the past, DBMS was able to only support Text/Numeric data.  
Nowadays, it is able to support:
- Images, Audios, Videos
- Spatial Data
- Time Series
- Data mining

## Centralized Database Environment
<a id="envs"></a>

Centralized Database Environment went through three stages:
1. **Mainframe environment:**
    - has a mainframe (containing database server and application server) and a group of connected monitors 
    - all processing occurs on one machine (the mainframe)
    - monitors connected using a dummy terminal
    - The terminal don't make any processing, they only send requests to mainframe to be processed. After processing, data is retrieved to be viewed by end-user. 
    - Limitations:
        - slow performance
        - single point of failure for both Database and Application
        
2. **Client/Server environment:**
    - two-tier environment; Database server and Client 
    - Client: is the application set up and installed on client side. The client in this case is called thick client as the application is set up and installed locally on every end-user machine. 
    - Application layer is no longer single point of failure.
    - Limitations:
        - Database is a single point of failure. 
        - High cost for support

3. **Internet Computing environment:**
    - three-tier; Database server/storage , application server , client
    - The client in this case is called thin client as it is considered a small application accessed through a browser. This facilitates maintenance and support.
    - Limitaions:
        - Application server is a single point of failure.
        - Database server is a single point of failure.
    - The Application tier has two types:
        - If 3-tier architecture then only one application server is used.
        - For n-tier architecture multiple paralled application servers are used.

## Distributed Database Environment
<a id="dis_envs"></a>

There are two methods to create a database environment:
- **Relpication:**
    - Full Relpication: the full database server is copied and pasted as another server.Both operate back to back. When the main server is down, the copied server starts working and all requests are rerouted to it.  
    - Partial Relpication: Part of the database is copied and set up which is enough to serve a certain branch/location.
- **Fragmentation:** 
    - the database is divided into fragments which are either:
        - horzontal as a group of records, or
        - vertical as a group of columns, or
        - hybrid from both

Through this the single point of failure is no longer an issue. The cost is, however, increased.

------------------------
------------------------

# Entity Relationship Diagram
<a id="ERD"></a> 

## Entity Relationship Modeling
<a id="erm"></a>

**Entity Relationship Modeling/Diagram:** is a method used in order to make the conceptual design.  
It identifies information required by the business by displaying the relevant entities and realtionships between them.

**Entity:** any independent existence in the system which may be described using a set of characteristics or attributes.

**Guidelines for designing Entity Relationship Modeling:**
- Entities to be described in the model
- Characteristics/attributes of those entities
- Can an attribute or a set of attributes be used to uniquely identify an etity
- Relationships between entities

## Entities and Attributes
<a id="entities&att"></a>

Representation:
- **Entities**  -> rectangle
- **Single/Simple Attribute** [attributes having only one value for a particular instance.] -> oval
- **Multi-valued Attribute** [attributes having a set of values for the same particular instance] -> double line oval
- **Composite Attribute** [attributes that can be divided to subparts.] -> connected ovals
- **Derived Attribute** [attributes that can calculated/made from existing attributes.] -> dotted oval


------------------------
------------------------
# ERD Mapping to Tables
<a id="ERD"></a> 