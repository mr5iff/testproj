CREATE DATABASE IF NOT EXISTS test_db;
USE test_db;

CREATE TABLE IF NOT EXISTS `piwik_track` (
  `time` datetime,
  `uid` varchar(256),
  `event_name` varchar(256)
) ENGINE=MyISAM;

DELIMITER $$
DROP PROCEDURE IF EXISTS prepare_data;
CREATE PROCEDURE prepare_data()
BEGIN
  DECLARE i INT DEFAULT 0;

  WHILE i < 1000000 DO
    INSERT INTO `piwik_track` (`time`, `uid`, `event_name`) VALUES (
      FROM_UNIXTIME(UNIX_TIMESTAMP('2017-03-01 01:00:00') + FLOOR(RAND()*31536000)),
      FLOOR(RAND()*1000),
      ELT(FLOOR(1 + RAND()*3), 'FIRST_INSTALL', 'COME_BACK', 'SOMETHING_ELSE')
    );
    SET i = i + 1;
  END WHILE;
END$$
DELIMITER ;

GRANT EXECUTE ON PROCEDURE test_db.prepare_data TO ''@'localhost';
flush privileges;

CALL prepare_data();