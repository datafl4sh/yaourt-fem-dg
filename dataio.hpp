

#ifdef WITH_SILO

#include <silo.h>

class silo_database
{
    DBfile          *m_siloDb;
    
public:
    silo_database()
        : m_siloDb(nullptr)
    {}
    
    bool create(const std::string& db_name)
    {
        m_siloDb = DBCreate(db_name.c_str(), DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
        if (m_siloDb)
            return true;
        
        std::cout << "Error creating database" << std::endl;
        return false;
    }
    
    bool open(const std::string& db_name)
    {
        m_siloDb = DBOpen(db_name.c_str(), DB_PDB, DB_APPEND);
        if (m_siloDb)
            return true;
        
        std::cout << "Error opening database" << std::endl;
        return false;
    }
    
    bool close()
    {
        if (m_siloDb)
            DBClose(m_siloDb);
        m_siloDb = NULL;
        return true;
    }
    
    ~silo_database()
    {
        if (m_siloDb)
            DBClose(m_siloDb);
    }
};

#endif
