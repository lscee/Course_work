{-| 
Module : Send
This module contains functions linking user and message
-}
module Send where

import Type
import Lib
import Control.Concurrent
import System.IO

-- |Send a message to user, record the userid into MVar
send :: MVar [Int] -> IO()
send mvar = do  
    --at random time intervals,
    -- print "interval"
    interval 

    --the thread should select one of the other users at random,
    -- print "randoming"
    r <- drawInt 1 10  -- random number for user pick
    m <- drawInt 1 10  -- random number for message pick
    let user = getUser r 

    --and send a random message to that user.
    --print "sending"
    let id = get_ID user
    let message = createMessage m id 
    print_mess message
    print_username user
    count_id id mvar
    print "Message sent"
    




