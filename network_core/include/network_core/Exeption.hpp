#pragma once

#ifndef NETWORK_EXEPTION_HPP_
#define NETWORK_EXEPTION_HPP_

#include <stdexcept>

namespace network {
  class NetworkError : public std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  class IOError : public NetworkError {
    using NetworkError::NetworkError;
  };

  /**
   * @brief Error for not existent file paths
   */
  class FileNotFoundError : public IOError {
    using IOError::IOError;
  };

  /**
   * @brief Error for not existent folder paths
   */
  class FolderNotFoundError : public IOError {
    using IOError::IOError;
  };

  /**
   * @brief Error thrown if some error occured during the parsing of the file
   */
  class ParseError : public IOError {
    using IOError::IOError;
  };

  /**
   * @brief Error thrown if not initialize variable
   */
  class NotInitializeError : public IOError {
    using IOError::IOError;
  };
} // namespace network
#endif // NETWORK_EXEPTION_HPP_