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
    - [Relationship](#Relationship)

- [ERD Mapping to Tables](#ERD_Mapping) 
    - [Mapping strong and weak entities](#Strong_Weak_Entities)
    - [Mapping of Relationship Types](#relationship_mapping)

- [Structured Query Language](#SQL) 
    - [Database Schema & Constraints](#Schema&Constraints)
    - [SQL - Data Definition Language](#DDL)
    - [SQL - Data Control Language](#DCL)  

- [Data Manipulation Language](#DML) 
    - [Insert Command](#insert)
    - [Update Command](#update)
    - [DELETE Command, Truncate](#delete)
    - [SELECT Command](#select)
    - [Comparison & Logical operators](#operators)
    - ["like" operators](#like)
    - [Alias](#alias)
    - [Order By](#OrderBy)
    - [Distinct](#distinct)
    - [Inner Join](#inner_j)
    - [Outer, Full Join](#outer_j)
    - [Self Join](#self_j)
    - [Max, Min, Count Functions](#max)
    - [Group By & Having](#group)

- [SQL - other DB objects](#other)
    - [Views](#view)
    - [Indexes](#indexes)

- [Normalization](#normalization)
    - [What is Normalization](#normalization_)
    - 


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


- **Entities**  -> rectangle
### Types of Attributes:
- **Single/Simple Attribute** [attributes having only one value for a particular instance.] -> oval
- **Multi-valued Attribute** [attributes having a set of values for the same particular instance] -> double line oval
- **Composite Attribute** [attributes that can be divided to subparts.] -> connected ovals
- **Derived Attribute** [attributes that can calculated/made from existing attributes.] -> dotted oval


**Weak Etities:** an entity that doesn't have a key attribute *and* is fully dependent on another entity. A double-lined rectangle is used to represent it.

## Relationship
<a id="Relationship"></a>

**Relationships**: a relationship is a connection between entity classes. For a relationship, three main parameters must be known:
- Degree
- Cardinality ratio
- Participation

### Degree
- refers to the number of entites existing within the relationship.
- represented by a diamond shape
- **Types**:
    - Binary: only two entities in the relationship
    - Unary/recursive: between the entity and itself.
    - Ternary: involving three entities.
 
### Cardinality ratio
- specifies the maximum number of relationships
- one to one, many to one, one to many, many to many
- for a ternary relationship
    - 3 binary relationships are assumed and a cardinality is given to each.
    - for the same side the cardinality must be the same. If not, then the design is to be changed to a closed loop.

### Participation
- specifies the minimum number of relationships instances that each entity can participate with.
- *double lines* for **must** and *single line* for **may**


Note that an attribute may be added to a relationship. 

------------------------
------------------------
# ERD Mapping to Tables
<a id="ERD_Mapping"></a>

## Mapping strong and weak entities
<a id="Strong_Weak_Entities"></a>

### Converting from Conceptual to logical design
1. Mapping regular entities
    - The attributes of an entity represent its columns
    - A primary key is set for the table. **Primary Keys:** must contain a unique value for each row of data **AND** CANNOT contain null values. If multiple attrubutes satisfy these conditions, the one with least storage is used.
    - For multi-valued attributes, a separate table is made and the primary key is used in that table as a foreign key.
    - Derived attributes aren't stored as they increase the storage taken. The only is exception is when that attribute is used frequently to retrieve the data. 
2. Mapping weak entities
    - primary key of owner entity is added as foreign key.
    - combination of the foreign key with another attribute offer a unique identification for the table.
 
## Mapping of Relationship Types
<a id="relationship_mapping"></a>

3. **One-to-many [binary/uniary]**: The primary key of the *one* entity is added as a foreign key in the *many*.
    - if the the relationship is unary then the name may differ in order to avoid confusion.
    - for weak entities where a combination of the foreign key with another attribute was used, no need to add a key as it's already there. 
4. **Many-to-many**: a new table is made having foreign keys from the main tables. Those keys form a unique combination.
5. **One-to-one**: 
    - May-May: The primary key of either could be add as foreign in the other or both could be added into a new table.
    - Must-Must: The two table are merged and either primary key could be chosen as the new primary key.
    - May-Must: The primary key of the *may* entity is added as a foreign key in the *must*.
6. **Mapping of ternary relationships**: a new table containing primary keys of entities is made. 

------------------------
------------------------

# Structured Query Language
<a id="SQL"></a>

## Database Schema & Constraints
<a id="Schema&Constraints"></a>

Structured Query Language (SQL) is the language used to interact with the database.  
It is divided into 3 main categories:
- DDL [Data Definition Language]
- DML [Data Manipulation Language]
- DCL [Data Control Language]

**Database Schema:** a schema is group of related objects in a database. There is one owner of the schema who has access to manipulate the structure of any object in the schema. 

**Database Constraints:** restrictions on the database which help in maintaining the data integrity. 
- **Primary Key**
- **Not Null**
- **Unique Key**
- **Referential Integrity [FK]:** Takes into consideration dealing with foreign key (Parent/Child records) and (inserting/deleting)
- **Check:** customized to suit the column

## SQL - Data Definition Language
<a id="DDL"></a>

Responsible for the structure of the database objects.
Used for creating/editing/deleting not for data manipulation.  

**Commands:**
- CREATE
    - CREATE TABLE TABLENAME (ColumnName DataType Constraint, Column2Name DataType Constraint)
    - **Ex:** CREATE TABLE Students (StudentID NUMBER PRIMARY KEY, FirstName CHAR(50) NOT NULL, LastName CHAR(50) BirthDate DATE)
- ALTER
    - ALTER TABLE TABLENAME ADD NewColumn DataType
    - ALTER TABLE TABLENAME DROP COLUMN ColumnName
- DROP
    - *removes Whole table*
    - DROP TABLE TableName
- TRUNCATE

## SQL - Data Control Language
<a id="DCL"></a>

Commands that gives access privilege to data. Privileges can be:
- System privilege
- **Object privilege**: includes persmission given to the user for the database objects.  

**Commands:**
- GRANT
    - GRANT COMMAND ON TABLE TableName TO UserName
    - **EX:**
        - GRANT SELECT ON TABLE Table1 TO UserA *userA is only allowed to select(view) the data*
        - GRANT ALL ON TABLE Table1 TO UserB, UserC *users B and C are allowed all DMLs*
        - GRANT SELECT ON TABLE Table1 TO UserA WITH GRANT OPTIONS *userA can view the data in addition to granting the permission to others*
- REVOKE
    - REVOKE COMMAND ON TABLE TableName FROM UserName


# Data Manipulation Language
<a id="DML"></a>

## Insert Command
<a id="insert"></a>

> INSERT INTO TableName (Col1_name, Col2_name, Col3_name, Col4_name) VALUES ('Col1_Value_char', 'Col2_Value_char', Col3_Value_Number, 'Col4_Value_Date')  
  
**OR** if columns order is known  

> INSERT INTO TableName VALUES ('Col1_Value_char', 'Col2_Value_char', Col3_Value_Number, 'Col4_Value_Date')

**OR** if not all columns have values
  
> INSERT INTO TableName (Col1_name, Col2_name, Col4_name) VALUES ('Col1_Value_char', 'Col2_Value_char', 'Col4_Value_Date')


## Update Command
<a id="update"></a>

Used for editing data already in the database.

> UPDATE Tablename  
> set column_to_be_updated = value  
> WHERE column_used_as_criteria = criteria  

**NOTE:** In case after *WHERE* is empty then value will be added to all records.

**Updating more than one column:**
> UPDATE Tablename  
> set column1 = value1 , column2 = value2  
> WHERE column_used_as_criteria = criteria

## DELETE Command, Truncate
<a id="delete"></a>

Deletes operates in record level.

- DELETE from Tablename
- where column = criteria

**DELETE vs TRUNCATE**  
Similar to *DELETE*, *TRUNCATE* is used for deleteing data however it operates on the table as whole unlike *DELETE* which operates on the record level. The table's data deleted however the table itself is not.
[*Truncate* operate like a *DELETE* without *WHERE*].  
Additionally, *TRUNCATE* command can't be rolled back unlike *DELETE* [unless commit is used] as DDLs auto commit. 

- TRUNCATE TABLE Tablename

## SELECT Command
<a id="select"></a>

Used mainly to view data

> SELECT COL1, COL2, [COL N]
> FROM Tablename
-------
**Note:** If colum name is placed in square brackets if the name contains spaces.  

**Coditional Selecting**  
> SELECT Col1, Col2
> FROM Tablename
> WHERE Col1 = value

## Comparison & Logical operators
<a id="operators"></a>

> SELECT *  
> from Tablename  
> WHERE Col1 > Val

**More than one conditions**

> SELECT *  
> from Tablename    
> WHERE Col1 > Upper_Val  
> AND COL1 < Lower_Val  

**And operator**

> SELECT *  
> from Tablename    
> WHERE Col1 between Upper_Val and Lower_Val

**Or operator**  
> SELECT Colname
> from Tablename    
> WHERE Col1 < Upper_Val  
> or Col2 < Lower_Val

>>**If values for same column**  
 
> SELECT Colname  
> from Tablename    
> WHERE Col1 In (Val1, Val2)  

*IN* is a multi-row operator meaning it deals with more than one value.

## "like" operator
<a id="like"></a>

The **like** operator is used to match pattern when the exacted wanted value is unkown. The unkown character is refered as **?**.

> SELECT *  
> from Tablename  
> WHERE fname like "Ahm?d"   
> OR fname like "?o*"

This query refers to either Ahmad/Ahmed or any name having *o* as the second character. * refers to zero or more characters.

## Alias
<a id="alias"></a>

> SELECT Col1, Col2 * 5.6 as NewCol  
> from Tablename

New column named *NewCol* will be returned, along selected columns, containing the written formula's values.

> Select Col1 + ' ' + Col2 as [Full Name]  
> from Tablename  
> Where Col3*12 > Value

**+** is used to concatenate columns Col1 and Col2 having a space in between.   
*Full Name* was written in square brackets as it contains space in its name.   



## Order By
<a id="OrderBy"></a>

> SELECT *  
> from Tablename  
> order by Col1, Col2 desc

*Default sorting order is ascendingly.*
**desc** is used to sort descendingly.


## Distinct
<a id="distinct"></a>

return unique values from specified column.

> Select distinct Col1   
> from Tablename

## Inner Join
<a id="inner_j"></a>

Used when the query is from more than one table.  

> SELECT Col1, Col2  
> FROM Table1name, Table2name  
> WHERE Condition_COl_Table1 = Condition_COl_Table2

**Note:**  
- Number of join condition is number of table -1
- The condition is usually between the primary key and foreign key.
- If the column names is the same for both table, then the column is preceeded by the table name: Table1name.Col1
- Aliasing could be used in order not to write the table's name

> SELECT Col1, Col2  
> FROM Table1name a, Table2name as b  
> WHERE a.Condition_COl = b.Consition_COl

**Using *Inner Join***
> SELECT Col1, Col2  
> FROM Table1name a inner join Table2name as b  
> on a.Condition_COl = b.Consition_COl

## Outer, Full Join
<a id="outer_j"></a>

There 3 types of OUTER JOIN:
- LEFT: returns full data from the left table
- RIGHT returns full data from the right table
- FULL: returns full data from both tables

> SELECT Col1, Col2  
> FROM Table1name a left outer join Table2name as b  
> on a.Condition_COl = b.Consition_COl

## Self Join
<a id="self_j"></a>

To join a table with itself a recursive relationship should be found. 

> SELECT a.Col1, b.Col1  
> FROM Tablename a, Tablename b  
> WHERE a.Col2 = b.Col1 

## Sub Queries
<a id="sub"></a>

> SELECT *  
> FROM Tablename  
> WHERE Col1 > (SELECT COL1 FROM Tablename WHERE COL2 = Value)

Other Multi-row operators: IN , ALL , ANY

## Max, Min, Count Functions
<a id="max"></a>

> SELECT MAX(Col1) as ColMAx, MIN(Col2) as ColMin  
> FROM Tablename

**Note:** Aggregate functions ignore null values. If Count is used it will skip null records.

## Group By & Having
<a id="group"></a>

> SELECT AVG(Col1)  
> FROM Tablename  
> GROUP BY Col2  
> HAVING MAX(Col1) > Value

*Having* is used when a condition is to be used with an aggregate function.

# SQL - Other DB objects
<a id="object"></a>

## Views
<a id="view"></a>

A view is **logical table** based on a table or another view.  

A view contains no table of its own, but is like a window through which data from table can be viewed or changed.

The table on which a view is based is called base table.

The view is stored as a SELECT statement in the data directory.

> CREATE VIEW viewname [Cols] /*If empty then as displayed in table */  
> AS /*Subquery to be written below*/
> SELECT Col1, Col2  
> FROM Tablename

**Note**: an extra line could be added after a **WHERE** clause in order to validate before a DML is done
> WITH CHECK OPTION

**TO edit anlreadey exiting view**  
> CREATE OR REPLACE VIEW viewname

**To delete a view**  
> DROP VIEW viewname

**Advantages of VIEW:**  
- Restrict data access.
- Make complex queries easy.
- Provide data independence.
- Present different views of the same data.

**Views Types**  

| **Feature** | Simple | Complex |
| :--- | :------: | :------: |
| **Number of Tables** | One | One or More |
| **Contain Functions** | No | Yes |
| **Contain Groups of Data** | No | Yes |
| **DML operations through a View** | Yes | Not always|


## Indexes
<a id="indexes"></a>

There are 2 problems in database. 
- Data is not sorted.
- Data is scattered in the physical memory. 

Indexes are used to speed up the retrieval of records in response to certain search conditions.  

- can be defined on one or more columns.  
- can be created by a user or by DBMS. DBMS usually creates index for the primary key.
- is used and maintained by DBMS. 

The index takes all the values of a column and sorts them. Then, beside each one of them, it adds a pointer to the location of each record in database. So, it looks like a minimized table with the columns's values ordered together, and beside each one of them there is a pointer to the address.

This is different from searching in the database object as a whole in the table where a full table scan is performed. Full Table Scan takes a lot of time, according to the table's size, number of columns and number of records it contains.  
However, when checking the index first it'll stop at a certain value and scanning will be end. This speeds up the search process.

On the other hand, while it speeds up the search, it slows down the DMLs. i.e. it makes Insert, Update and Delete a bit slower. The information is mentioned in 2 places [the table of database & the index as an object]. Thus, when updating a certain value, both loactions need to be changed. Then the index should be resorted.

Guidelines for creating an index when:
- data retrieval occurs often.
- the columns are used in many search conditions, or in Join conditions, where DBMS adds an additional index to the primary key.
- the column has a large number of nulls (check only the records with values).

Do not create an index when:
- an index when the table is updated frequently.

> CREATE INDEX Index_name ON Tablename (Column)

To remove this index:  
> DROP INDEX Index_name.

# Normalization
<a id="normalization"></a>

## What is Normalization
<a id="normalization_"></a>

It's a process that takes a table through a series of tests(Normal Forms) in order to do one of two things:
- certifying the goodness of a design. minimizing the redundancy or some anomalies; i.e. the problems that can happen such as: Insert, Update or Delete.
- to have a new design for the database. 

Why do we need
Normalization? In this example,
assuming that is a design
for a real database, we can find that
the design is consisted of 2 tables.
One of them is called Employee Department
and the other is called Employee Project.
The common part between the 2 tables
is that SSN of the employee is mentioned
in Employee Project table as a foreign key
Also, it forms with the project number;
both of them together forming
a Composite Primary Key. What are the problems
that may face us here? It's clear in
this example that there is a problem of
redundancy; i.e. repetition. What's meant by
Redundancy here? When I say that
a certain employee in a certain department,
I mention the department's name and
number of the department's manager.
For example, in the department no. 5
there are 100 employees. This means
I wrote the department's name 100 times
and I wrote number of the department's manager
100 times. It's the same here
in the table of Employee with Project.
Here is the column of the Employee Name.
Data of the employee is already
recorded in Employee Department table.
However, to record that an employee
has worked in a certain project
I record the name of this employee
in addition to the project's name with
its location. Therefore, as much as employees
have worked in a certain project,
the project's name has been repeated
and its location has also been repeated.
These are the redundancy problems.
Redundancy causes many problems.
One of its problems is that it affects
the storage, the tables' size,
the database performance in general.
Other problems that may happen
for such a design
may be like Insert, Update
or Delete anomalies. What does this mean?
Here, the primary key of
the table is SSN.
So, the master of this data is the employee.
Which means that I cannot insert
data of any department with no employees
working in it. It's the same
with the projects; I cannot
enter data of the project without
inserting number of the employee
working in this project. These are the Insert problems.
Assuming that I want to delete.
A certain employee,
say the last employee here,
has resigned. So, I decided to delete
his record. I delete using
the primary key. What is the primary key
here? it's SSN.
So deleting this employee
means that this department is empty.
i.e. it's not exist any more.
This department has this employee
only. As he has resigned,
and with this design of data,
so this department is closed.
This is a problem with Delete. There is
a problem with Update. We've mentioned
that when a certain employee works
in the department, say, no. 5,
I should record the department's name
and number of the department's manager.
Let's assume that the manager of this department
was changed. If there are, say,
100 employees, and I need to write
an update statement to edit
number of this department's manager,
This update statement, instead of editing
one record, it should edit
100 records. This will slow
down the database performance.
These are the problems that
may be found in such design.
Therefore, Normalization helps
to avoid duplication of data
or Redundancy,
Insert, Delete and Update anomalies
which are the problems that can
happen with Update, Insert and Delete,
Frequent non values: back to
our example, when the department's manager
is changed or currently there is
no manger instead of him yet
if I write in his record "Null",
suddenly I'll have 100 Null
in this column only.
So, this design also
causes a problem in the existence
of Null values. Again,
When do we use Normalization?
1- To test the goodness of a design;
confirming that the design
has no problems.
2- In case of an old database,
or trying to build a system based on existing files
I can use Normalization
for creating a design.