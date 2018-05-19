import firebase from 'firebase';
import { Actions } from 'react-native-router-flux';
import {
  EMAIL_CHANGED,
  PASSWORD_CHANGED,
  LOGIN_USER_SUCCESS,
  LOGIN_USER_FAIL,
  LOGIN_USER
} from './types';

// typeを定数にしておくと、エラーで気がつきやすくなる。
export const emailChanged = (text) =>{
  return {
    type: EMAIL_CHANGED,
    payload: text
  };
};

export const passwordChanged = (text) =>{
  return {
    type: PASSWORD_CHANGED,
    payload: text
  };
};

export const loginUser = ({ email, password }) => {
  console.log('in loginUser action', email, password);

  // return function, thunk will see. thunk consider to execute
  return (dispatch) => {

    dispatch({type: LOGIN_USER});

    firebase.auth().signInWithEmailAndPassword(email, password)
    .then(user => loginUserSuccess(dispatch, user))
    .catch((error)=> {
      console.log(error) //やっとくこと、多分firebaseのエラー全般。
      firebase.auth().createUserWithEmailAndPassword(email, password)
        .then(user => loginUserSuccess(dispatch, user))
        .catch(( )=> loginUserFail(dispatch));
    });
    };
  };


const loginUserFail = (dispatch) => {
  dispatch({
    type: LOGIN_USER_FAIL
  })
}

const loginUserSuccess = (dispatch, user) => {
  dispatch({
    type: LOGIN_USER_SUCCESS,
    payload: user
  });

  Actions.main();
  // keyの値がそのままmethod名になる
};




// action creators are function and must return function
// actoin must has type property

// OR

// action creator with redux-thunk
// function will be called with dispatchEvent
