select count(distinct T1.`uid`)
from piwik_track T1
where 
T1.`event_name` = 'FIRST_INSTALL' and
T1.`time` like '2017-04-01%' and
exists (
  select 1
  from piwik_track T2
  where 
  (T2.`time` between '2017-04-02 00:00:00' and '2017-04-08 23:59:59') and
  T1.`uid` = T2.`uid`
);