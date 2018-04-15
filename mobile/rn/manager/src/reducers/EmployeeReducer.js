import {
  EMPLOYEES_FETCH_SUCCESS
} from '../actions/types';

const INITIAL_STATE = {};

export default ( state = INITIAL_STATE, action) => {
  console.log('export default ( state = INITIAL_STATE, action)', action);
  switch (action.type){
    case EMPLOYEES_FETCH_SUCCESS:
      console.log('start case EMPLOYEES_FETCH_SUCCESS');
      console.log(action);
      return action.payload;
    default:
      return state;
  }
};
