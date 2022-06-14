module Main where

import Lib
import Type
import Thread
import Control.Concurrent
import Control.Monad
import System.IO
import Send

main :: IO ()
main =  do 
    m <- newMVar [0,0,0,0,0,0,0,0,0,0]
    replicateM 10 $ createThread m
    threadDelay 10000000
    print "sheeeeeesh"
    user_count <- takeMVar m
    sumMessage user_count 100
    print_result user_count


 
  
