import { useAuth0 } from '@auth0/auth0-react';
import React, { useCallback } from 'react';
import { getEnv } from './helpers';

const GetPermissionButton = () => {
  const { getAccessTokenSilently } = useAuth0();
  const getPermission = useCallback(() => {
    const getAccessToken = async () => {
      const audience = getEnv('REACT_APP_AUDIENCE');
      const accessToken = await getAccessTokenSilently({
        audience: audience,
      });
      console.log(accessToken);
    };
    getAccessToken();
  }, [getAccessTokenSilently]);
  return <button onClick={() => getPermission()}>Get Permission</button>;
};

export default GetPermissionButton;
