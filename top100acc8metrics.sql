-------------------create external table------------
CREATE EXTERNAL TABLE IF NOT EXISTS jin.top_account_201708 (
  accountid string,
  accountname string
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
   'separatorChar' = ',',
   'quoteChar' = '\"',
   'escapeChar' = '\\'
   )
STORED AS TEXTFILE
LOCATION 's3://athena-us/datausage/accountRef/'
tblproperties ("skip.header.line.count"="1");


------------all metric for top100 accounts----------------------------------------------
--top100 account id--
with top100 as (
SELECT * FROM jin.top_account_201708 limit 100
),
-- mapdata metrics
t1 as (
with y as 
(SELECT accountid,date_trunc('hour',stoptime) AS hourstamp,
CASE
    WHEN firstopcodedesc = 'sendAuthenticationInfo' THEN 'countSAI'
    WHEN firstopcodedesc = 'updateLocation' THEN 'countUL'
    WHEN firstopcodedesc = 'updateGprsLocation' THEN 'countUGL'
    WHEN firstopcodedesc = 'cancelLocation' THEN 'countCL'
    --roll up on firstopcodedec will generate null for all code and selected hour
    WHEN firstopcodedesc is null THEN 'mapsum' 
END as metric,
count(1) value
FROM jin_20170904_111836.mapdata
where firstopcodedesc is not null and accountid in (select accountid from top100) 
and month in (5,6,7,8)
GROUP BY accountid,2,rollup(firstopcodedesc)
)
select * from y where metric!=''
),
-- create Pdp v1 and create Pdp v2 sum metric
t2 as(
SELECT accountid,date_trunc('hour',stoptime) AS hourstamp,
'createPdpCountV1' as metric,
SUM(CASE WHEN causecode= 128 AND recordtype = 1 THEN 1 ELSE 0 END) AS value
FROM jin_20170904_111836.gtpv1 
where accountid in (select accountid from top100) and month in (5,6,7,8)
GROUP BY 1,2,3
union
SELECT accountid,date_trunc('hour',stoptime) AS hourstamp,
'createPdpCountV2' as metric,
SUM(CASE WHEN causecode= 128 AND recordtype = 1 THEN 1 ELSE 0 END) AS value
FROM jin_20170904_111836.gtpv2
where accountid in (select accountid from top100) and month in (5,6,7,8)
GROUP BY 1,2,3
),
-- imsi number metric-----
t3 as(
SELECT accountid,date_trunc('hour',stoptime) AS hourstamp,
'imsiSum' as metric,count(distinct(imsi)) value
FROM jin_20170904_111836.gtpv1
where accountid in (select accountid from top100) and month in (5,6,7,8)
GROUP BY 1,2,3
)
select accountid,hourstamp,extract(DAY_OF_WEEK FROM hourstamp) as day_of_week,metric,value from 
(select * from t1 union select * from t2 union select * from t3) t_all
order by hourstamp,accountid,metric
-----------------all metric end line-----------------------------