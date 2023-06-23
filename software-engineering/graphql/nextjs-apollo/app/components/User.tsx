"use client";

import { useSubscription, useMutation } from "@apollo/client";
import { gql } from "../../apollo/__generated__/client";
import { useState } from "react";

const ADD_USER = gql(`mutation AddUser($name: String!) {
  addUser(name: $name) {
    name
  }
}`)

const SUBSCRIPTION_ALL_USERS = gql(`subscription SUBSCRIBE_ALL_USERS {
  users {
    name
  }
}`)

const User = () => {
  const [userName, setUserName] = useState('');
  const [addUser] = useMutation(ADD_USER);
  const { data, loading, error } = useSubscription(SUBSCRIPTION_ALL_USERS);
  console.log({ data, loading, error })

  const handleAddUser = async () => {
    try {
      await addUser({ variables: { name: userName } });
      setUserName('');
    } catch (err) {
      console.error(err);
    }
  };

  if (loading) {
    return <div>読み込み中</div>;
  }
  return (
    <div>
      {error && <div>{error.message}</div>}
      <ul>
        {data?.users?.map((v, i) => <li key={String(i)}>{v.name}</li>)}
      </ul>
      <input
        type="text"
        value={userName}
        onChange={(e) => setUserName(e.target.value)}
      />
      <button onClick={handleAddUser}>Add User</button>
    </div>
  );
};

export default User;
