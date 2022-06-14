{-# LANGUAGE CPP #-}
{-# LANGUAGE NoRebindableSyntax #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
{-# OPTIONS_GHC -Wno-missing-safe-haskell-mode #-}
module Paths_lsc (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where

import qualified Control.Exception as Exception
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude

#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []
bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "/Users/lsc/.cabal/bin"
libdir     = "/Users/lsc/.cabal/lib/x86_64-osx-ghc-8.10.7/lsc-0.1.0.0-inplace"
dynlibdir  = "/Users/lsc/.cabal/lib/x86_64-osx-ghc-8.10.7"
datadir    = "/Users/lsc/.cabal/share/x86_64-osx-ghc-8.10.7/lsc-0.1.0.0"
libexecdir = "/Users/lsc/.cabal/libexec/x86_64-osx-ghc-8.10.7/lsc-0.1.0.0"
sysconfdir = "/Users/lsc/.cabal/etc"

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "lsc_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "lsc_libdir") (\_ -> return libdir)
getDynLibDir = catchIO (getEnv "lsc_dynlibdir") (\_ -> return dynlibdir)
getDataDir = catchIO (getEnv "lsc_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "lsc_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "lsc_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "/" ++ name)
