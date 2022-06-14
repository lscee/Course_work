{-| 
Module : Thread
This module containing functions about the threads
-}
module Thread where

import Control.Concurrent
import Control.Monad
import System.IO
import System.Random
import Send

-- |Create new Thread using forkIO
createThread :: MVar [Int] -> IO()   
createThread m= do
    forkIO $ do
        send m 
        send m 
        send m 
        send m 
        send m 
        send m 
        send m 
        send m 
        send m 
        send m
    putStrLn "processing"