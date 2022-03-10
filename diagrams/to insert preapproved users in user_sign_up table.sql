/*
use this concat formula in excel to create insertion values
	=CONCAT(", ('",TEXTJOIN("', '",FALSE,A2:F2),"')")

*/
USE Funds360;
GO

DROP TABLE IF EXISTS #tmp_data;
GO
SELECT	LTRIM(RTRIM(t.Name)) + ' ' + LTRIM(RTRIM(t.Surname)) AS USER_name, LTRIM(RTRIM(t.Email)) AS login_id, 1 AS is_preapproved --
	  , CONCAT(
			'{"firstName": "'
		  , LTRIM(RTRIM(t.Name))
		  , '", "lastName": "'
		  , LTRIM(RTRIM(t.Surname))
		  , '", "userName": "'
		  , LTRIM(RTRIM(t.Email))
		  , '", "companyName": "'
		  , LTRIM(RTRIM(t.Company))
		  , '", "[Manager/Advisor]": "'
		  , LTRIM(RTRIM(t.[Manager/Advisor]))
		  , '", "positionDetail": "'
		  , LTRIM(RTRIM(t.Title))
		  , '"}') AS other_details, 2 AS activation_status, 1 AS user_accepted
INTO	#tmp_data
FROM	(	VALUES
				----


---- Paste the User details created from excel below
 ('Katy', 'Gavelas', 'katy.gavelas@360fundinsight.com', 'Test', 'Manager', 'Director')








----
) t (Name, Surname, Email, Company, [Manager/Advisor], Title);





---- insert into main table 
BEGIN
	INSERT INTO dbo.user_sign_ups (user_name, login_id, is_preapproved, other_details, activation_status, user_accepted)
	SELECT			t.USER_name, t.login_id, t.is_preapproved, t.other_details, t.activation_status, t.user_accepted
	FROM			#tmp_data t
		LEFT JOIN	dbo.user_sign_ups u ON t.login_id = u.login_id
	WHERE			u.login_id IS NULL;
END;


---- update token for the users
BEGIN
	DECLARE @PassphraseEnteredByUser NVARCHAR(256) = N'support@valuefy.com';
	DECLARE @add_authenticator INT = 1;
	DECLARE @authenticator VARBINARY(256) = 1;

	DECLARE @activation_status INT = 3;
	DECLARE @user_accepted INT = 1;

	----DECLARE @token VARCHAR(1000) = CONVERT(VARCHAR(1000), ENCRYPTBYPASSPHRASE(@PassphraseEnteredByUser, @login_id, @add_authenticator, @authenticator), 1);
	----SELECT @token;
	----SELECT	CONVERT(VARCHAR(MAX) --
	----			  , DECRYPTBYPASSPHRASE(@PassphraseEnteredByUser, CONVERT(VARBINARY(1000), @token, 1), @add_authenticator, @authenticator));

	---- update token for all users
	BEGIN
		UPDATE		u
		SET			u.token = CONVERT(VARCHAR(1000), ENCRYPTBYPASSPHRASE(@PassphraseEnteredByUser, u.login_id, @add_authenticator, @authenticator), 1) --
				  , u.activation_status = @activation_status, u.status_update_time = GETDATE(), u.user_accepted = ISNULL(@user_accepted, 0)
		FROM		dbo.user_sign_ups u
			JOIN (	 SELECT DISTINCT   login_id FROM   #tmp_data) t ON u.login_id = t.login_id
		WHERE		u.activation_status = 2
					AND u.is_preapproved = 1;
	END;
END;


---- get latest inserted data for the preapproved users
SELECT		u.login_id, CONCAT('https://fundsfairway.com/#/user/register-trial?utoken=', u.token) AS Registration_URL
FROM		dbo.user_sign_ups u
	JOIN	#tmp_data t ON t.login_id = u.login_id
WHERE		u.is_preapproved = 1
			AND u.activation_status = 3
			AND u.status_update_time >= DATEADD(MINUTE, -5, GETDATE());



------ get already registered ones
--SELECT	login_id, CONCAT('https://fundsfairway.com/#/user/register-trial?utoken=', token) AS Registration_URL
--FROM	dbo.user_sign_ups
--WHERE	login_id IN ( 'Izaskun.biurrunecheverria@seguroseci.es', 'mpaz@grupounicaja.es' );

