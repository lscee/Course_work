{-| 
Module : Type
This module contains 
    -data type definition 
    -constant values
    -functions about new data
-}
module Type where

import Control.Concurrent

-- |user data definition
data User = User{
    username :: String,
    id_user :: Int
}deriving (Eq,Show)

-- |message data definition
data Message = Message{
    text :: String,
    reciever :: Int
}deriving (Eq,Show)

-- |constant value 
usernames = ["lsc","paul","Maria","Jesus","Romma","Vineus","Pompeii","Finland","Haskell","Vatican"]
userids = [1..10]

-- |Get user id
get_ID :: User -> Int
get_ID(User {id_user = a})= a
-- |Get user name
get_name :: User -> String
get_name (User {username = u})= u
--Create a new user
getUser :: Int -> User
getUser id =  User { 
    username= last $ take id usernames, 
    id_user= last $ take id userids
    }

-- |create a new message
createMessage :: Int -> Int-> Message
createMessage r id = Message{
    text = "hello " ++ (last $ take r usernames) ,
    reciever = id
 }
    
-- |Count message for different user 
count_id :: Int -> MVar [Int] -> IO()
count_id id mvar = do
    temp <- takeMVar mvar
    let part =  (temp!!(id-1) +1) : reverse (take (length temp -id ) (reverse temp))
    let new =  take (id-1) temp ++ part
    putMVar mvar new
    
-- |Output the result
print_result :: [Int] -> IO()
print_result result = do
    print result
    putStrLn("user1  recived " ++  (show $ result!!0)++ " messges")
    putStrLn("user2  recived " ++  (show $ result!!1)++ " messges")
    putStrLn("user3  recived " ++  (show $ result!!2)++ " messges")
    putStrLn("user4  recived " ++  (show $ result!!3)++ " messges")
    putStrLn("user5  recived " ++  (show $ result!!4)++ " messges")
    putStrLn("user6  recived " ++  (show $ result!!5)++ " messges")
    putStrLn("user7  recived " ++  (show $ result!!6)++ " messges")
    putStrLn("user8  recived " ++  (show $ result!!7)++ " messges")
    putStrLn("user9  recived " ++  (show $ result!!8)++ " messges")
    putStrLn("user10 recived " ++  (show $ result!!9)++ " messges")

-- |Print Message text
print_mess :: Message -> IO()
print_mess message  = do
    print "message content:"
    putStrLn (text message)

-- |Print user name
print_username :: User -> IO()
print_username user = do
    print "user name:"
    putStrLn (username user)

-- |Check all messages send
sumMessage :: [Int] -> Int ->IO()
sumMessage result total
    | sum result ==total =do print "ALL MESSAGES ARE SENT!"
    | otherwise = do print "Missing Message!"