{-| 
Module : Lib
This is a default module containing irrelavant functions 
-}
module Lib where
    
import System.IO
import Control.Concurrent
import System.Random
import Type

-- |Generate one random number in certain range
drawInt :: Int -> Int -> IO Int
drawInt x y = getStdRandom (randomR (x,y))

-- |Get a random interval
interval :: IO()
interval = do
    r <- drawInt 1 10  
    let random_interval = r * 100000
    --print random_interval
    threadDelay random_interval