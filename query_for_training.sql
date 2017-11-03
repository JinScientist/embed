with months as (
select (cast(month(current_date-interval '4' month) as INTEGER),cast(year(current_date-interval '4' month) as INTEGER))
union all
select (cast(month(current_date-interval '3' month) as INTEGER),cast(year(current_date-interval '3' month) as INTEGER))
union all
select (cast(month(current_date-interval '2' month) as INTEGER),cast(year(current_date-interval '2' month) as INTEGER))
union all
select (cast(month(current_date-interval '1' month) as INTEGER),cast(year(current_date-interval '1' month) as INTEGER))
),
top100 as (
SELECT * FROM refdata.top_account_201708 limit 100
),
t1 as (
with y as 
(SELECT accountid,date_trunc('hour',stoptime) AS hourstamp,
CASE
    WHEN firstopcodedesc = 'sendAuthenticationInfo' THEN 'countSAI'
    WHEN firstopcodedesc = 'updateLocation' THEN 'countUL'
    WHEN firstopcodedesc = 'updateGprsLocation' THEN 'countUGL'
    WHEN firstopcodedesc = 'cancelLocation' THEN 'countCL'
    WHEN firstopcodedesc is null THEN 'mapsum' 
END as metric,
count(1) value
FROM signaling_20171018_112152.mapdata
where firstopcodedesc is not null and accountid in (select accountid from top100) 
and (month,year) in (select * from months)
GROUP BY accountid,2,rollup(firstopcodedesc)
)
select * from y where metric!=''
),
t2 as(
SELECT accountid,date_trunc('hour',stoptime) AS hourstamp,
'createPdpCountV1' as metric,
SUM(CASE WHEN causecode= 128 AND recordtype = 1 THEN 1 ELSE 0 END) AS value
FROM signaling_20171018_112152.gtpv1 
where accountid in (select accountid from top100) and (month,year) in (select * from months)
GROUP BY 1,2,3
union
SELECT accountid,date_trunc('hour',stoptime) AS hourstamp,
'deletePdpCountV1' as metric,
SUM(CASE WHEN causecode= 128 AND recordtype = 2 THEN 1 ELSE 0 END) AS value
FROM signaling_20171018_112152.gtpv1 
where accountid in (select accountid from top100) and (month,year) in (select * from months)
GROUP BY 1,2,3
),
t3 as(
SELECT accountid,date_trunc('hour',stoptime) AS hourstamp,
'imsiSum' as metric,count(distinct(imsi)) value
FROM signaling_20171018_112152.gtpv1
where accountid in (select accountid from top100) and (month,year) in (select * from months)
GROUP BY 1,2,3
)
select accountid,hourstamp,extract(DAY_OF_WEEK FROM hourstamp) as day_of_week,metric,value from 
(select * from t1 union select * from t2 union select * from t3) t_all
order by hourstamp,accountid,metric