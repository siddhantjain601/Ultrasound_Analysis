import tuple from '../db.js';

export const createUser = async (uname, email, hpasswd) => {
  const result = await tuple.query(
    'INSERT INTO users (name, email, password) VALUES ($1, $2, $3) RETURNING *', //checking eith the 3 parameters in input
    [uname, email, hpasswd]
  );
  return result.rows[0]; 
};

export const findUserByEmail = async (email) => {
  const result = await tuple.query('SELECT * FROM users WHERE email = $1', [email]); //checking with the first parametetr
  return result.rows[0]; 
};
